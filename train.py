"""CPU LoRA: default eight-update benchmark, or --full for three training epochs.

Every invocation starts from the downloaded base model with new adapters.
Full training uses validation loss to select a checkpoint; neither mode reads test data.
"""

import argparse
import hashlib
import json
import random
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import psutil
import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from download_model import sha256
from config import MODEL_DIR, PROJECT_ROOT, SYSTEM_PROMPT, TRAIN_PATH, CPU_THREADS


SEED = 20260910
LEARNING_RATE = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
MAX_GRAD_NORM = 1.0
MAX_SEQUENCE_LENGTH = 256


def make_batch(tokenizer, record):
    """Convert one complete conversation into inputs and answer-only labels."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=record["schema"])},
        {"role": "user", "content": record["question"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    conversation = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": record["sql"]}],
        tokenize=False, add_generation_prompt=False,
    )
    if not conversation.startswith(prompt):
        raise ValueError("Conversation does not start with the expected prompt")
    encoded = tokenizer(conversation, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    answer_start = next(index for index, (_, end) in enumerate(offsets) if end > len(prompt))
    if offsets[answer_start][0] != len(prompt):
        raise ValueError("A token crosses the prompt/answer boundary; review masking")
    if len(encoded["input_ids"]) > MAX_SEQUENCE_LENGTH:
        raise ValueError("Example exceeds the benchmark length limit; do not silently truncate")

    # Outer list adds the batch dimension: [sequence] becomes [1, sequence].
    batch = {name: torch.tensor([values], dtype=torch.long) for name, values in encoded.items()}
    labels = batch["input_ids"].clone()
    labels[:, :answer_start] = -100
    if answer_start == 0 or not (labels[:, 1:] != -100).any():
        raise ValueError("Expected both a masked prompt and scored answer tokens")
    batch["labels"] = labels
    return batch


def frozen_fingerprint(model):
    """Hash every frozen parameter without copying the whole model."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            # Our CPU float32 tensors expose their bytes through a NumPy view.
            digest.update(memoryview(parameter.detach().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def memory_snapshot():
    """Windows peak working set is resident memory, not all committed memory."""
    info = psutil.Process().memory_info()
    return {
        "resident_gib": info.rss / 2**30,
        "peak_resident_gib": getattr(info, "peak_wset", info.rss) / 2**30,
        "system_available_gib": psutil.virtual_memory().available / 2**30,
    }


def probe_loss(model, batch):
    """Compare the same training example before/after; this is not validation."""
    model.eval()
    with torch.no_grad():
        value = model(**batch).loss.item()
    model.train()
    return value


def benchmark_main():
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    initial_memory = memory_snapshot()
    records = [json.loads(line) for line in TRAIN_PATH.read_text(encoding="utf-8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    examples = [(record, make_batch(tokenizer, record)) for record in records]
    families = sorted({record["query_family"] for record in records})
    # Use the longest example in each family, chosen without looking at model loss.
    selected = [max(
        (item for item in examples if item[0]["query_family"] == family),
        key=lambda item: item[1]["input_ids"].shape[1],
    ) for family in families]
    schedule = [selected[0]] + selected  # One warm-up update, then seven measured updates.

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = PROJECT_ROOT / "results" / f"lora_benchmark_{run_id}"
    output_dir.mkdir(parents=True)
    print(f"Output: {output_dir}", flush=True)
    print("Loading base weights on CPU in float32...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.float32,
    ).to("cpu")
    model.config.use_cache = False  # Training uses complete sequences, not generation's KV cache.

    # PEFT inserts A/B matrices and freezes the original parameters.
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(model, config)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Only LoRA parameters should be trainable")
    model.print_trainable_parameters()
    before_adapters = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    before_frozen = frozen_fingerprint(model)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=LEARNING_RATE, weight_decay=0.0)
    before_loss = probe_loss(model, selected[0][1])
    step_results = []

    # This is the core SFT training loop. Batch size = 1; no gradient accumulation.
    model.train()
    with (output_dir / "steps.jsonl").open("w", encoding="utf-8") as log:
        for step, (record, batch) in enumerate(schedule, 1):
            started = perf_counter()
            optimizer.zero_grad(set_to_none=True)       # Clear the previous step's gradients.
            output = model(**batch)                    # Forward pass, including masked cross-entropy.
            loss = output.loss                         # Transformers shifts next-token labels internally.
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()                            # Compute gradients; does not update weights.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(trainable.values()), MAX_GRAD_NORM, error_if_nonfinite=True,
            )
            if grad_norm.item() == 0:
                raise ValueError("No training gradient reached the adapters")
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError("A frozen parameter unexpectedly has a gradient")
            optimizer.step()                           # Apply one update to the adapter parameters.
            seconds = perf_counter() - started
            result = {
                "step": step, "warmup": step == 1, "example_id": record["id"],
                "family": record["query_family"], "tokens": batch["input_ids"].shape[1],
                "scored_tokens": (batch["labels"][:, 1:] != -100).sum().item(),
                "loss_before_update": loss.item(), "gradient_norm_before_clipping": grad_norm.item(),
                "seconds": seconds, **memory_snapshot(),
            }
            step_results.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
            print(f"Step {step}/{len(schedule)}: loss={loss.item():.4f}, "
                  f"gradient norm={grad_norm.item():.3f}, {seconds:.2f}s", flush=True)
            del output, loss  # Release this step's outputs before starting the next one.

    optimizer.zero_grad(set_to_none=True)
    after_loss = probe_loss(model, selected[0][1])
    frozen_unchanged = frozen_fingerprint(model) == before_frozen
    changed = sum(not torch.equal(parameter.detach(), before_adapters[name])
                  for name, parameter in trainable.items())
    if not frozen_unchanged or changed == 0:
        raise ValueError("Expected changed adapters and unchanged frozen base weights")
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    expected = get_peft_model_state_dict(model)
    if saved.keys() != expected.keys() or any(not torch.equal(saved[key], expected[key]) for key in saved):
        raise ValueError("Saved adapter tensors do not match the trained adapter")

    times = [item["seconds"] for item in step_results if not item["warmup"]]
    median_seconds = statistics.median(times)
    model_manifest = json.loads((MODEL_DIR / "download_manifest.json").read_text())
    report = {
        "purpose": "benchmark_only_start_full_training_fresh", "run_id": run_id,
        "model_id": model_manifest["model_id"], "revision": model_manifest["revision"],
        "seed": SEED, "device": "cpu", "dtype": "float32", "cpu_threads": CPU_THREADS,
        "batch_size": 1, "updates": len(schedule), "unique_examples": len(selected),
        "train_examples": len(records), "training_file_sha256": sha256(TRAIN_PATH),
        "script_sha256": sha256(PROJECT_ROOT / "train.py"), "system_prompt": SYSTEM_PROMPT,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "psutil")},
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": 0.0,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0, "max_grad_norm": MAX_GRAD_NORM, "gradient_checkpointing": False,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "changed_adapter_tensors": changed, "adapter_tensors": len(trainable),
        "frozen_weights_unchanged": frozen_unchanged, "saved_adapter_tensors_verified": True,
        "probe_example_id": selected[0][0]["id"],
        "probe_loss_before": before_loss, "probe_loss_after": after_loss,
        "median_step_seconds": median_seconds, "measured_step_range_seconds": [min(times), max(times)],
        "estimated_epoch_minutes": median_seconds * len(records) / 60,
        "estimated_three_epoch_minutes": median_seconds * len(records) * 3 / 60,
        "estimate_caveat": "Short run; excludes validation, saving, startup, and sustained CPU throttling",
        "initial_memory": initial_memory, "final_memory": memory_snapshot(),
        "steps": step_results,
    }
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSame training example loss: {before_loss:.4f} -> {after_loss:.4f}")
    print(f"Adapter tensors changed: {changed}/{len(trainable)}; frozen weights unchanged: {frozen_unchanged}")
    print(f"Median measured step: {median_seconds:.2f}s")
    print(f"Estimated training-only time: {report['estimated_epoch_minutes']:.1f} min/epoch; "
          f"{report['estimated_three_epoch_minutes']:.1f} min for 3 epochs")
    print(f"Peak resident process memory: {report['final_memory']['peak_resident_gib']:.2f} GiB")
    print(f"Saved benchmark report and verified adapter: {output_dir}")


def validation_loss(model, examples):
    """Mean loss per answer token, not an unweighted mean of example losses."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    details = []
    try:
        with torch.no_grad():
            for record, batch in examples:
                tokens = (batch["labels"][:, 1:] != -100).sum().item()
                value = model(**batch).loss.item()
                if tokens == 0 or not torch.isfinite(torch.tensor(value)):
                    raise ValueError("Invalid validation targets or loss")
                total_loss += value * tokens
                total_tokens += tokens
                details.append({"id": record["id"], "loss": value, "scored_tokens": tokens})
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Validation set is empty")
    return {"loss": total_loss / total_tokens, "scored_tokens": total_tokens, "examples": details}


def save_adapter_verified(model, path):
    """Save an inference checkpoint, then verify its tensors against memory."""
    model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    saved = load_file(str(path / "adapter_model.safetensors"))
    expected = get_peft_model_state_dict(model)
    if saved.keys() != expected.keys() or any(not torch.equal(saved[key], expected[key]) for key in saved):
        raise ValueError("Checkpoint tensor verification failed")


def full_main(epochs):
    """Train on every training example; select the lowest validation-loss epoch."""
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    rng = random.Random(SEED)
    started_run = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    validation_path = PROJECT_ROOT / "data" / "validation.jsonl"

    def load_examples(path):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not records:
            raise ValueError(f"Empty dataset: {path}")
        return [(record, make_batch(tokenizer, record)) for record in records]

    examples = load_examples(TRAIN_PATH)
    validation = load_examples(validation_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = PROJECT_ROOT / "results" / f"sft_v1_{run_id}"
    output_dir.mkdir(parents=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Fresh adapters; {len(examples)} train / {len(validation)} validation; {epochs} epochs", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.float32,
    ).to("cpu")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none",
    ))
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Only adapters may be trainable")
    model.print_trainable_parameters()
    before_frozen = frozen_fingerprint(model)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=LEARNING_RATE, weight_decay=0.0)
    manifest = json.loads((MODEL_DIR / "download_manifest.json").read_text())
    report = {
        "status": "running", "run_id": run_id, "seed": SEED, "epochs_planned": epochs,
        "train_examples": len(examples), "validation_examples": len(validation),
        "model_id": manifest["model_id"], "revision": manifest["revision"],
        "device": "cpu", "dtype": "float32", "cpu_threads": CPU_THREADS, "batch_size": 1,
        "learning_rate": LEARNING_RATE, "optimizer": "AdamW", "weight_decay": 0.0,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"], "max_grad_norm": MAX_GRAD_NORM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH, "gradient_accumulation_steps": 1,
        "system_prompt": SYSTEM_PROMPT, "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "input_sha256": {path.name: sha256(path) for path in (TRAIN_PATH, validation_path)},
        "script_sha256": sha256(PROJECT_ROOT / "train.py"),
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        "selection_rule": "Lowest answer-token-weighted validation loss; earliest epoch wins ties",
        "checkpoint_type": "Adapter inference checkpoints, not optimizer/resume checkpoints",
        "initial_memory": memory_snapshot(), "epochs": [],
    }
    report_path = output_dir / "training.json"

    def write_report():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_report()
    print("Measuring untrained validation loss...", flush=True)
    baseline = validation_loss(model, validation)
    report["baseline_validation_loss"] = baseline["loss"]
    (output_dir / "validation_before.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    write_report()
    print(f"Baseline validation loss: {baseline['loss']:.6f}", flush=True)
    best_loss = float("inf")
    best_epoch = None
    step = 0
    with (output_dir / "steps.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            order = list(range(len(examples)))
            rng.shuffle(order)  # Every example once per epoch, in a new seeded order.
            model.train()
            training_total, training_tokens = 0.0, 0
            epoch_started = perf_counter()
            for position, index in enumerate(order, 1):
                record, batch = examples[index]
                started = perf_counter()
                optimizer.zero_grad(set_to_none=True)
                output = model(**batch)
                loss = output.loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(trainable.values()), MAX_GRAD_NORM, error_if_nonfinite=True,
                )
                optimizer.step()
                step += 1
                tokens = (batch["labels"][:, 1:] != -100).sum().item()
                training_total += loss.item() * tokens
                training_tokens += tokens
                log.write(json.dumps({"step": step, "epoch": epoch, "example_id": record["id"],
                                      "loss": loss.item(), "scored_tokens": tokens,
                                      "gradient_norm": grad_norm.item(),
                                      "seconds": perf_counter() - started}) + "\n")
                log.flush()
                if position == 1 or position % 25 == 0 or position == len(order):
                    print(f"Epoch {epoch}/{epochs}, example {position}/{len(order)}, "
                          f"loss={loss.item():.5f}, elapsed={perf_counter() - epoch_started:.0f}s", flush=True)
                del output, loss
            optimizer.zero_grad(set_to_none=True)
            train_seconds = perf_counter() - epoch_started
            print(f"Epoch {epoch}: checking validation loss...", flush=True)
            measured = validation_loss(model, validation)
            checkpoint = output_dir / f"epoch_{epoch}"
            save_adapter_verified(model, checkpoint)
            (output_dir / f"validation_epoch_{epoch}.json").write_text(
                json.dumps(measured, indent=2), encoding="utf-8",
            )
            if measured["loss"] < best_loss:
                best_loss, best_epoch = measured["loss"], epoch
            report["epochs"].append({"epoch": epoch, "updates": step,
                                     "training_loss": training_total / training_tokens,
                                     "validation_loss": measured["loss"], "train_seconds": train_seconds,
                                     "checkpoint": checkpoint.name, "memory": memory_snapshot()})
            report.update(best_epoch=best_epoch, best_validation_loss=best_loss)
            write_report()
            print(f"Epoch {epoch}: validation={measured['loss']:.6f}; best epoch={best_epoch}", flush=True)

    if frozen_fingerprint(model) != before_frozen:
        raise ValueError("Frozen base weights changed")
    # Load the selected checkpoint from disk as a separate adapter, not just live epoch-3 weights.
    best_path = output_dir / f"epoch_{best_epoch}"
    model.load_adapter(str(best_path), adapter_name="selected", is_trainable=False)
    model.set_adapter("selected")
    reloaded = validation_loss(model, validation)
    if abs(reloaded["loss"] - best_loss) > 1e-6:
        raise ValueError("Reloaded checkpoint did not reproduce its validation loss")
    report.update(status="complete", updates=step, best_checkpoint=best_path.name,
                  selected_better_than_base=best_loss < baseline["loss"],
                  frozen_weights_unchanged=True, checkpoint_reload_verified=True,
                  reloaded_validation_loss=reloaded["loss"],
                  elapsed_seconds=perf_counter() - started_run, final_memory=memory_snapshot())
    write_report()
    print(f"\nComplete: {step} updates; selected {best_path}")
    print(f"Validation loss: base {baseline['loss']:.6f} -> selected {best_loss:.6f}")
    print("This is loss-based checkpoint selection, not SQL execution accuracy. Test data was not read.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Train all examples and select using validation loss")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs in full mode (default: 3)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.full:
        full_main(args.epochs)
    else:
        benchmark_main()
