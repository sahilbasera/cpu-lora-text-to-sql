"""Generate SQL for one pilot question using the local, unmodified model."""

import json
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from config import MODEL_DIR, PROJECT_ROOT, SYSTEM_PROMPT, MAX_NEW_TOKENS, CPU_THREADS


PILOT_PATH = PROJECT_ROOT / "data" / "pilot.jsonl"


def main() -> None:
    # Pick a fixed example by file order, before seeing any model output.
    with PILOT_PATH.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["query_family"] == "country_amount_gt":
                break
        else:
            raise ValueError("No combined-filter example found in the pilot data")

    torch.set_num_threads(CPU_THREADS)
    print(f"Example: {record['id']}", flush=True)
    print(f"Question: {record['question']}", flush=True)
    print("Loading local model on CPU (float32)...", flush=True)
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.float32,
    ).to("cpu")
    model.eval()
    load_seconds = perf_counter() - started

    # Only instructions and the question go into the generation prompt.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=record["schema"])},
        {"role": "user", "content": record["question"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]

    # Greedy decoding: pick the most likely next token without random sampling.
    settings = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    print("Generating SQL...", flush=True)
    started = perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=settings)
    generation_seconds = perf_counter() - started

    # generate() returns the prompt followed by the answer; keep only the answer.
    answer_ids = output[0, prompt_length:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    eos_ids = settings.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    stopped_on_eos = bool(len(answer_ids)) and answer_ids[-1].item() in eos_ids

    print(f"\nMODEL OUTPUT:\n{answer}")
    print(f"\nREFERENCE SQL (not supplied to the model):\n{record['sql']}")
    print(f"\nLoad time: {load_seconds:.2f} seconds")
    print(f"Generation time: {generation_seconds:.2f} seconds")
    print(f"Prompt tokens: {prompt_length}; generated tokens: {len(answer_ids)}")
    print(f"Generation rate: {len(answer_ids) / generation_seconds:.2f} tokens/second")
    print(f"Stop reason: {'end token' if stopped_on_eos else 'token limit'}")
    print(f"Runtime: torch {torch.__version__}; CPU; float32; {CPU_THREADS} threads")
    print("This is an inference demonstration, not an execution-based accuracy score.")


if __name__ == "__main__":
    main()
