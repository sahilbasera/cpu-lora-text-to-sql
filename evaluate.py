"""Evaluate the local base model or a saved LoRA adapter using SQLite results."""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from download_model import sha256


ROOT = Path(__file__).resolve().parent
MAX_ROWS = 1000
MAX_SQL_CHARACTERS = 8000
QUERY_SECONDS = 0.5
MAX_PROGRESS_CALLS = 100  # SQLite invokes the callback every 1,000 VM instructions.


def execute_readonly(path, sql):
    """Run one SELECT against orders, with restricted access and bounded work."""
    if len(sql) > MAX_SQL_CHARACTERS or not re.match(r"\s*SELECT\b", sql, re.I):
        raise ValueError("Expected one plain SELECT statement")

    def authorize(action, table, column, database, source):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if (action == sqlite3.SQLITE_READ and database == "main"
                and table == "orders" and column in ("id", "country", "amount", "")):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    deadline = perf_counter() + QUERY_SECONDS
    calls = 0

    def limit_work():
        nonlocal calls
        calls += 1
        return calls >= MAX_PROGRESS_CALLS or perf_counter() > deadline

    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only = ON")
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 100_000)
        db.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        db.set_authorizer(authorize)
        db.set_progress_handler(limit_work, 1000)
        # execute(), unlike executescript(), rejects multiple statements.
        cursor = db.execute(sql)
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchmany(MAX_ROWS + 1)
        if len(rows) > MAX_ROWS:
            raise ValueError("Result row limit exceeded")
        return columns, rows


def score_sql(sql, references):
    """Require the expected columns and row multiset on every fixture."""
    checks = []
    for path, (expected_columns, expected_rows) in references.items():
        try:
            columns, rows = execute_readonly(path, sql)
            columns_match = columns == expected_columns
            rows_match = Counter(rows) == Counter(expected_rows)
            checks.append({
                "fixture": path.name, "correct": columns_match and rows_match,
                "columns_match": columns_match, "rows_match": rows_match,
                "expected_count": len(expected_rows), "actual_count": len(rows),
                "missing_rows": list((Counter(expected_rows) - Counter(rows)).elements())[:5],
                "extra_rows": list((Counter(rows) - Counter(expected_rows)).elements())[:5],
            })
        except (sqlite3.Error, ValueError) as error:
            checks.append({"fixture": path.name, "correct": False, "error": str(error)})
    correct = bool(checks) and all(check["correct"] for check in checks)
    status = "correct" if correct else (
        "execution_error" if any("error" in check for check in checks) else "result_mismatch"
    )
    return {"correct": correct, "status": status, "fixtures": checks}


def main(split="pilot", adapter=None):
    # Heavy imports stay here so evaluator tests do not need to load a model.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from config import CPU_THREADS, MAX_NEW_TOKENS, MODEL_DIR, SYSTEM_PROMPT

    pilot_path = ROOT / "data" / f"{split}.jsonl"
    records = [json.loads(line) for line in pilot_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    fixtures = [ROOT / "data" / "fixtures" / name for name in manifest["fixtures"]]
    if len(records) != manifest["splits"][split] or len(fixtures) != 3:
        raise ValueError("Dataset counts do not match the manifest")
    # Reference errors abort the run, rather than being counted as model errors.
    references = {
        record["id"]: {path: execute_readonly(path, record["sql"]) for path in fixtures}
        for record in records
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = ROOT / "results" / f"{'lora' if adapter else 'base'}_{split}_{run_id}"
    output_dir.mkdir(parents=True)
    print(f"Saving results to {output_dir}", flush=True)
    torch.set_num_threads(CPU_THREADS)
    print("Loading local Qwen on CPU...", flush=True)
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.float32,
    ).to("cpu")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True)
    model.eval()
    load_seconds = perf_counter() - started
    settings = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True,
    )
    model_manifest = json.loads((MODEL_DIR / "download_manifest.json").read_text())
    metadata = {
        "run_id": run_id, "split": split, "model_id": model_manifest["model_id"],
        "adapter": str(adapter.resolve()) if adapter else None,
        "adapter_sha256": {name: sha256(adapter / name) for name in
                           ("adapter_config.json", "adapter_model.safetensors")} if adapter else None,
        "revision": model_manifest["revision"], "system_prompt": SYSTEM_PROMPT,
        "generation_settings": settings.to_dict(), "device": "cpu", "dtype": "float32",
        "cpu_threads": CPU_THREADS, "model_load_seconds": load_seconds,
        "versions": {name: version(name) for name in ("torch", "transformers", "tokenizers", "peft")},
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [pilot_path, *fixtures]},
        "evaluator_sha256": sha256(Path(__file__)),
        "scoring": "Plain SELECT only; exact column names/order and unordered row multisets on all fixtures",
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results = []
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output_file:
        for index, record in enumerate(records, 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(schema=record["schema"])},
                {"role": "user", "content": record["question"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            prompt_length = inputs["input_ids"].shape[1]
            started = perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=settings)
            seconds = perf_counter() - started
            answer_ids = generated[0, prompt_length:]
            raw_answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
            sql = raw_answer.strip()  # No fence removal or SQL repair before scoring.
            eos_ids = settings.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            stopped = bool(len(answer_ids)) and answer_ids[-1].item() in eos_ids
            result = {
                "id": record["id"], "query_family": record["query_family"],
                "question": record["question"], "reference_sql": record["sql"],
                "raw_answer": raw_answer, "generated_sql": sql,
                "prompt_tokens": prompt_length, "generated_tokens": len(answer_ids),
                "generation_seconds": seconds, "stop_reason": "end_token" if stopped else "token_limit",
                "exact_string_match": sql == record["sql"],
                **score_sql(sql, references[record["id"]]),
            }
            results.append(result)
            output_file.write(json.dumps(result) + "\n")
            output_file.flush()  # Preserve completed examples if the run is interrupted.
            print(f"[{index:02}/{len(records)}] {record['id']}: {result['status']} ({seconds:.2f}s)", flush=True)

    groups = defaultdict(list)
    for result in results:
        groups[result["query_family"]].append(result)
    correct = sum(result["correct"] for result in results)
    summary = {
        "total": len(results), "correct": correct, "accuracy": correct / len(results),
        "statuses": dict(Counter(result["status"] for result in results)),
        "exact_string_matches": sum(result["exact_string_match"] for result in results),
        "token_limit_count": sum(result["stop_reason"] == "token_limit" for result in results),
        "generation_seconds": sum(result["generation_seconds"] for result in results),
        "by_family": {
            family: {"correct": sum(item["correct"] for item in items), "total": len(items),
                     "accuracy": sum(item["correct"] for item in items) / len(items)}
            for family, items in sorted(groups.items())
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nExecution accuracy: {correct}/{len(results)} ({summary['accuracy']:.1%})")
    for family, counts in summary["by_family"].items():
        print(f"  {family}: {counts['correct']}/{counts['total']}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["pilot", "validation", "test"], default="pilot")
    parser.add_argument("--adapter", type=Path, help="Local saved adapter directory; omit for base model")
    args = parser.parse_args()
    if args.adapter and not (args.adapter / "adapter_model.safetensors").is_file():
        parser.error("Adapter directory must contain adapter_model.safetensors")
    main(args.split, args.adapter)
