"""Compare paired base/adapter evaluation results after checking experiment parity."""

import argparse
import json
from collections import Counter
from pathlib import Path


def compare(base_dir, lora_dir):
    base_meta = json.loads((base_dir / "run.json").read_text())
    lora_meta = json.loads((lora_dir / "run.json").read_text())
    for field in ("split", "model_id", "revision", "system_prompt", "generation_settings",
                  "device", "dtype", "cpu_threads", "versions", "input_sha256",
                  "evaluator_sha256", "scoring"):
        if base_meta[field] != lora_meta[field]:
            raise ValueError(f"Comparison settings differ: {field}")
    if base_meta["adapter"] is not None or lora_meta["adapter"] is None:
        raise ValueError("Expected base results first, adapter results second")

    def read_results(directory):
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        summary = json.loads((directory / "summary.json").read_text())
        if len(rows) != summary["total"] or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Incomplete or duplicate results")
        return {row["id"]: row for row in rows}, summary

    base, base_summary = read_results(base_dir)
    lora, lora_summary = read_results(lora_dir)
    if base.keys() != lora.keys():
        raise ValueError("Example IDs differ")
    counts = Counter()
    changes = []
    for key, before in base.items():
        after = lora[key]
        for field in ("question", "reference_sql", "query_family"):
            if before[field] != after[field]:
                raise ValueError(f"Example differs: {key}/{field}")
        category = ("both_correct" if after["correct"] else "regression") if before["correct"] else (
            "fixed" if after["correct"] else "both_wrong")
        counts[category] += 1
        if category != "both_correct":
            changes.append({"id": key, "category": category, "question": before["question"],
                            "reference_sql": before["reference_sql"],
                            "base_sql": before["generated_sql"], "lora_sql": after["generated_sql"]})
    report = {
        "base_run": str(base_dir.resolve()), "lora_run": str(lora_dir.resolve()),
        "settings_match": True, "total": len(base),
        "base_correct": base_summary["correct"], "lora_correct": lora_summary["correct"],
        "accuracy_change_percentage_points": 100 * (lora_summary["accuracy"] - base_summary["accuracy"]),
        "paired_outcomes": dict(counts), "changes_and_remaining_errors": changes,
        "base_by_family": base_summary["by_family"], "lora_by_family": lora_summary["by_family"],
    }
    destination = lora_dir / "comparison.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_run", type=Path)
    parser.add_argument("lora_run", type=Path)
    args = parser.parse_args()
    compare(args.base_run, args.lora_run)
