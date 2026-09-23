# From the benchmark to full SFT

Run full training with `.venv\Scripts\python.exe train.py --full --epochs 3`. Without `--full`, the script still runs the short benchmark. Every invocation starts with the original Qwen weights and freshly initialized adapters.

## What changed

The benchmark checked whether learning worked and measured speed using eight updates. Full mode uses every one of the 196 training examples once per epoch. Three epochs therefore produce 588 optimizer updates, at batch size one. No gradient accumulation is used.

The original forward/backward/update mechanics are unchanged. Rank 8, alpha 16, query/value targets, learning rate 0.0001, float32, and four CPU threads are also unchanged. Three epochs is our initial experiment choice, not an established optimum. No test-set data is read.

## One epoch

```python
order = list(range(len(examples)))
rng.shuffle(order)
```

An epoch is one pass through the full training dataset. The order is shuffled each epoch with a seeded random generator. This changes the order, not membership: all 196 examples occur exactly once in each epoch.

For each example we run:

```python
optimizer.zero_grad(set_to_none=True)
output = model(**batch)
loss = output.loss
loss.backward()
torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()
```

The snippet abbreviates the actual parameter list and safety checks. Clear gradients, calculate loss, compute gradients, clip them, and apply one parameter update. Those same operations repeat 588 times.

## Validation is measurement, not more training

Before any updates, and after each epoch, `validation_loss` measures the 56 validation examples. It uses the same answer-only loss mask as training, but:

```python
model.eval()
with torch.no_grad():
    value = model(**batch).loss.item()
```

There is no `backward()` or `optimizer.step()` during validation. Evaluation mode and disabled gradient tracking are separate controls. The function restores the previous model mode afterward.

Validation includes the correct answers to calculate teacher-forced next-token loss. This is different from asking the model to generate SQL without seeing the answer and checking its execution. Lower validation loss is useful for checkpoint selection but does not prove higher SQL execution accuracy.

## How we average loss

Answers have different lengths, and each example's loss is already averaged over its scored answer tokens. To report one loss across the dataset, we weight each example by that token count:

```python
total_loss += example_loss * scored_tokens
total_tokens += scored_tokens
dataset_loss = total_loss / total_tokens
```

For losses 2 and 4 on answers containing 1 and 3 scored tokens, the result is `(2*1 + 4*3)/4 = 3.5`, not 3. Our tests verify this aggregation.

The reported training loss is accumulated while the model changes throughout an epoch. Validation loss is measured with a single fixed checkpoint after the epoch. They are not interchangeable measurements.

## Checkpoints and selection

At each epoch end, we save that epoch's adapter weights and configuration. Each saved tensor is read back and compared with the live adapter state.

```python
if validation_loss < best_loss:
    best_loss = validation_loss
    best_epoch = epoch
```

We select the trained checkpoint with the lowest validation loss. A tie keeps the earlier epoch. The pre-training validation score remains in the report; if none of the trained checkpoints beats it, the report says so rather than claiming improvement.

The last epoch is not automatically the selected checkpoint. Check `best_checkpoint` in `training.json`.

These are adapter inference checkpoints. They do not contain AdamW optimizer state and are not exact-resume checkpoints. They require the original base model for inference.

## Verification at completion

We compare the fingerprint of all frozen base parameters with their pre-training fingerprint. The values must be unchanged.

Then we load the selected adapter from disk into the same unchanged base model as a separate adapter, activate it, and remeasure validation loss. It must reproduce the selected checkpoint's score within 0.000001. This checks saved-adapter loading and selection; it does not load a second copy of the entire base model.

## Files saved for a full run

- `training.json`: settings, versions, data hashes, baseline and epoch losses, selected checkpoint, timings, and verification status.
- `steps.jsonl`: one record per update, including example ID, loss, gradient norm, and timing.
- `validation_before.json`: per-example loss before training.
- `validation_epoch_1.json` and corresponding later files: per-example validation losses.
- `epoch_1/`, `epoch_2/`, `epoch_3/`: saved adapters and their configurations.

Run folders are unique, so the pilot and benchmark results remain available. Status is initially `running` and becomes `complete` only after the selected adapter passes reload verification. If interrupted, completed epoch adapters and flushed step logs remain, but the final verification has not happened.

## What this establishes

A completed run establishes that SFT updated adapters across the full training split and identifies a checkpoint by held-out validation loss. Version 1 subsequently compared generated SQL from the base and selected adapted model under identical settings: 63/84 versus 70/84. See the [results index](../results/README.md) for the final comparison and its limitations.
