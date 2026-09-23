# Final version 1: base versus LoRA

The original Qwen2.5-0.5B-Instruct model scored 63/84 (75.0%). With the epoch-1 adapter selected previously by validation loss, it scored 70/84 (83.3%): a net gain of seven questions, or 8.33 percentage points.

The paired comparison contains 59 both-correct, 11 fixed, 4 regressed, and 10 both-wrong examples. Both runs used the same test questions, references, fixtures, prompt, greedy generation configuration, model revision, software versions, CPU settings, and evaluator. `compare_results.py` checked these metadata fields and example identities. Correctness requires matching column names/order and row multisets across all three fixtures; this is not exact SQL string matching.

| Family | Base | LoRA |
|---|---:|---:|
| Country equality | 8/12 | 12/12 |
| Greater than | 12/12 | 12/12 |
| At least | 7/12 | 7/12 |
| Less than | 7/12 | 7/12 |
| At most | 5/12 | 8/12 |
| Country and amount | 12/12 | 12/12 |
| Inclusive range | 12/12 | 12/12 |

## What improved

Eight fixes restored all expected columns: four country questions had selected only `amount`, and four maximum-amount questions had selected only `id, country`. LoRA produced `SELECT *`, matching the dataset's output convention. This should be understood partly as learning that convention, not as eight new filter-reasoning successes; natural-language requests do not always explicitly spell out the required columns.

Three further fixes handled "not above" correctly as `<=`, replacing incorrect `NOT BETWEEN` expressions.

## Regressions and remaining failures

All four regressions concern "equal to or below": the base used `<=`, while LoRA generated `>=`.

Five "not below" examples remain wrong: LoRA generates `<=` rather than `>=`. Five "does not reach" examples remain wrong: LoRA generates `!=` rather than `<`. These repeated phrasings expose systematic language-to-operator problems despite very low validation loss.

## Interpretation and limits

The observed improvement is real under the fixed evaluation contract for this 84-question synthetic test, but is not evidence of broad text-to-SQL mastery or uniformly better reasoning. The test contains repeated templates with different values, so these are not 84 independent linguistic challenges. This was one training seed and one schema, not a statistical robustness study.

The earlier 41/42 pilot score used a different set of phrasings. The fair improvement comparison is 63/84 versus 70/84 on the same test questions, not pilot versus test.

No checkpoint, prompt, data, or scoring adjustments were made in response to these test results. This test has now been inspected; if its errors inform version 2 development, use a newly reserved final test for independent assessment. Preserve this version 1 result as recorded.

Full predictions are saved in this directory and in `../base_test_20260923T051755848526Z`. `comparison.json` lists every fixed, regressed, and remaining incorrect example with both SQL outputs.
