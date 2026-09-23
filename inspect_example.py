"""Show how one training example becomes tokens and masked next-token targets."""

from __future__ import annotations

import json
from typing import Any

from transformers import AutoTokenizer


from config import MODEL_DIR, TRAIN_PATH, SYSTEM_PROMPT
EXAMPLE_FAMILY = "country_amount_gt"
PROMPT_ROWS_TO_SHOW = 6

def load_example() -> dict[str, Any]:
    """Load the first combined-filter example from the training split."""
    with TRAIN_PATH.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["query_family"] == EXAMPLE_FAMILY:
                return record
    raise ValueError(f"No {EXAMPLE_FAMILY!r} example found in {TRAIN_PATH}")


def visible_token(tokenizer: Any, token_id: int) -> str:
    """Render one token so spaces, newlines, and control tokens are visible."""
    text = tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return repr(text)


def main() -> None:
    """Format, tokenize, mask, and display one example."""
    record = load_example()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=record["schema"])},
        {"role": "user", "content": record["question"]},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": record["sql"]}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    formatted_chat = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    if not formatted_chat.startswith(prompt_text):
        raise ValueError("The formatted prompt is not a prefix of the conversation")

    encoded = tokenizer(
        formatted_chat, add_special_tokens=False, return_offsets_mapping=True
    )
    full_ids = encoded["input_ids"]
    answer_character = len(prompt_text)
    assistant_start = next(
        index
        for index, (_, end) in enumerate(encoded["offset_mapping"])
        if end > answer_character
    )

    labels = full_ids.copy()
    labels[:assistant_start] = [-100] * assistant_start

    print("EXAMPLE")
    print(f"ID:       {record['id']}")
    print(f"Question: {record['question']}")
    print(f"Answer:   {record['sql']}")
    print("\nFORMATTED CONVERSATION")
    print(formatted_chat)

    learned_targets = len(full_ids) - assistant_start
    print("TOKEN COUNTS")
    print(f"Full conversation: {len(full_ids)}")
    print(f"Masked prefix:     {assistant_start}")
    print(f"Learned targets:   {learned_targets}")

    first_target = max(1, assistant_start - PROMPT_ROWS_TO_SHOW)
    print("\nNEXT-TOKEN TRAINING VIEW")
    print(f"{'POS':>4}  {'INPUT TOKEN':<24} {'TARGET TOKEN':<24} LOSS")
    print("-" * 66)
    for target_position in range(first_target, len(full_ids)):
        input_position = target_position - 1
        input_token = visible_token(tokenizer, full_ids[input_position])
        target_token = visible_token(tokenizer, full_ids[target_position])
        contributes = "YES" if labels[target_position] != -100 else "no"
        print(
            f"{input_position:>4}  {input_token:<24.24} "
            f"{target_token:<24.24} {contributes}"
        )

    print("\n-100 means that target is ignored by the loss function.")
    print("The first YES row predicts the first SQL token from the assistant header.")
    print("The assistant end token is also a learned target.")


if __name__ == "__main__":
    main()
