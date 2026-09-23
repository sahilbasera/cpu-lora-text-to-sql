# Pilot rerun after range wording clarification

The unchanged base model passed 41/42 questions (97.6%), compared with 38/42 (90.5%) in base_pilot_20260904T235125761604Z. All six inclusive-range questions now pass. The three original ID-range interpretations disappeared after the questions explicitly specified amounts. This is a wording correction, not an SFT improvement; the evaluation inputs changed.

The remaining failure is unchanged: "Show orders worth at most 25." produces `SELECT * FROM orders WHERE amount LIMIT 25` instead of `SELECT * FROM orders WHERE amount <= 25;`.

Five range templates were clarified, affecting six pilot, eight training, zero validation, and eight test questions. All other record fields, including SQL targets, values, IDs, and split assignments, were verified unchanged. The previous dataset is preserved in `results/data_before_range_clarification_20260910`. Both evaluation runs are preserved separately.

The model revision, system prompt, generation configuration, runtime package versions, CPU settings, evaluator hash, and fixture hashes match the original run. Twelve data/evaluator tests passed before inference. Total generation time was 93.21 seconds, excluding startup, model loading, and scoring. No output reached the token limit.

This is a narrow, near-ceiling pilot baseline with one observed error. It supports proceeding with the version 1 learning exercise, but does not promise a substantial accuracy improvement from SFT. No model training or model evaluation on the final test split occurred. The test wording was corrected as part of the template review, without consulting test predictions.
