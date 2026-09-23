# Base pilot interpretation

The unmodified Qwen2.5-0.5B-Instruct model passed 38 of 42 questions (90.5%) under the recorded evaluator. Each pass requires matching columns and row multisets on all three fixtures. Total generation time was 83.18 seconds, excluding startup, model loading, and SQL checks. No response reached the token limit.

One clear failure: `pilot_amount_le_006`, "Show orders worth at most 25.", produced `SELECT * FROM orders WHERE amount LIMIT 25` instead of filtering `amount <= 25`.

Three mismatches share an ambiguous template: `pilot_amount_range_001`, `002`, and `003` ask "List orders from ... through ..." without specifying amount. The model interpreted these as ID ranges. These are reference mismatches, but not clean evidence of a model error. The earlier wording cleanup missed this template. Preserve this run as the original result; clarify the range templates and record any rerun separately before drawing training conclusions.

Only 1 of 42 outputs exactly matched the reference string, while 38 matched execution results. Formatting differences such as omitted final semicolons explain why string equality is not the primary metric.

This run uses only pilot questions. It does not establish test accuracy or an SFT improvement. The evaluator is scoped to this simple SELECT/filter dataset: it checks column names/order, ignores row order, rejects functions and non-plain-SELECT output, and enforces query work and result-size limits. It is not a general SQL equivalence checker. No automatic SQL repair or code-fence stripping was applied.
