# Understanding our first SFT training script

The script is `train.py`. By default it performs eight actual optimizer updates, then stops. That mode is a benchmark. The newer `--full --epochs 3` mode trains on the full training split and uses validation loss to select a checkpoint; see `FULL_TRAINING_WALKTHROUGH.md`. Every invocation loads the original local model and initializes fresh LoRA adapters. It never resumes the previous adapter automatically.

Read the core loop in `benchmark_main` first. The surrounding code prepares its inputs and checks its effects. The line numbers below describe the original benchmark script and may shift as it evolves.

## What happened in the first run

- Model: Qwen2.5-0.5B-Instruct on CPU, float32, four CPU threads.
- Eight updates across seven unique training examples. The first example is used for both a warm-up update and a measured update.
- Batch size one, no gradient accumulation, no gradient checkpointing.
- 540,672 trainable parameters out of 494,573,440 total, including adapters: about 0.1093%.
- All 96 adapter tensors changed. Every frozen parameter had the same byte fingerprint before and after training.
- Median measured step: 1.40 seconds; measured range 1.17–1.62 seconds, excluding the warm-up.
- Estimated training-only time: 4.6 minutes for 196 updates (one epoch), or 13.7 minutes for three epochs at the same settings. This short measurement excludes validation, startup, saving, and possible sustained CPU throttling. Three epochs is an illustration, not a selected training duration.
- Peak resident process memory: 3.16 GiB. This is the Windows working-set peak, not total system RAM usage or all committed virtual memory.
- The same training example's loss fell from 0.0639 to 0.00133. This demonstrates fitting a training example; it does not establish improved SQL accuracy on unseen questions.

Results are in `results/lora_benchmark_20260911T041959985110Z`.

## Imports and settings: lines 7–30

`torch` handles tensors, automatic differentiation, and the optimizer. `transformers` loads Qwen and its tokenizer. `peft` supplies LoRA layers and adapter saving. `psutil` measures process/system memory. `safetensors` reads the saved adapter for verification. The remaining imports handle files, timings, hashes, and reports.

We reuse the model path, training path, and system prompt from `config.py` so the training task uses the same instructions as inference.

- `SEED`: controls PyTorch randomness, including adapter initialization. It is not a guarantee of bit-identical results across hardware/library changes.
- `CPU_THREADS = 4`: how many CPU threads PyTorch may use for its tensor operations.
- `LEARNING_RATE = 1e-4`: 0.0001. AdamW uses it to scale its adaptive updates. It does not mean every weight changes by exactly 0.0001.
- `LORA_RANK = 8`: the inner dimension of each adapter's A/B pair.
- `LORA_ALPHA = 16`: standard LoRA scales the adapter contribution by alpha/rank, here 2.
- `MAX_GRAD_NORM = 1.0`: clips unusually large overall adapter gradients before an update.
- `MAX_SEQUENCE_LENGTH = 256`: the benchmark fails if an example exceeds this length. It does not silently cut off the answer or pad every example to 256.

## Preparing an example: `make_batch`, lines 33–61

1. Construct system and user messages from the schema and question.
2. Format a prompt that ends at the assistant header.
3. Format the complete conversation by adding the reference SQL as the assistant response.
4. Verify that the complete conversation starts with that prompt.
5. Tokenize the complete conversation once. Character offsets identify exactly where the SQL begins in token space.
6. Reject a token that overlaps the prompt/answer boundary, rather than silently making an approximate mask.
7. Turn the lists into integer tensors with an outer batch dimension.
8. Copy the input IDs to labels and mask the prompt targets with -100.

```python
labels = batch["input_ids"].clone()
labels[:, :answer_start] = -100
batch["labels"] = labels
```

The colon means all examples in the batch; we currently have only one. `:answer_start` selects all token positions before the SQL.

An illustrative batch looks like:

```text
input_ids:      [[prompt IDs ... | SQL IDs ... | ending IDs]]
attention_mask: [[1, 1, 1, ...   | 1, 1, ...  | 1, 1      ]]
labels:         [[-100, ...      | SQL IDs ... | ending IDs]]
```

The attention mask contains ones because this one-example batch needs no padding. It does not replace the model's built-in causal attention mask. The loss mask is stored in `labels`. The scored suffix includes the SQL, `<|im_end|>`, and the template's final newline.

During training the correct answer is in the input: causal attention prevents future-token peeking, and the loss aligns each prediction with the following token. During generation the reference answer is absent.

## Measurement helpers: lines 64–91

`frozen_fingerprint` hashes all frozen parameters in memory before and after training. Equal hashes are evidence that those parameter values did not change. It uses views of CPU float32 tensors instead of storing a second copy of the whole base model.

`memory_snapshot` reports resident process memory, Windows peak working set, and available system memory.

`probe_loss` evaluates the same selected training example before and after the updates. `model.eval()` selects evaluation behavior, and `torch.no_grad()` disables gradient tracking for this measurement. It restores training mode afterward. Neither function freezes parameters by itself.

## Selecting the benchmark examples: lines 94–109

We read all 196 training records and prepare their input tensors, but we do not update on all of them. For each question family we choose the longest tokenized example, without inspecting model performance. This samples the seven operations with relatively long inputs.

`schedule = [selected[0]] + selected` adds one warm-up update before the seven measured updates. That warm-up really changes the adapters. Its time is excluded from the runtime estimate, but it is included in the total of eight updates.

This selection is for a timing/mechanics check. Full training will use all training examples, ordinarily shuffled, from fresh base weights and fresh adapters.

## Loading and adding LoRA: lines 111–130

`from_pretrained` constructs Qwen and loads its saved parameters. `local_files_only=True` makes the model load local. Float32 specifies parameter precision; `.to("cpu")` selects the device.

`model.config.use_cache = False` disables the incremental KV cache used in generation. Here we process complete training sequences.

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
)
model = get_peft_model(model, config)
```

`CAUSAL_LM` identifies next-token language modeling. `q_proj` and `v_proj` are the attention query and value projections. We attach adapters to those projections in each of Qwen's 24 layers. We are not replacing attention or implementing a new Qwen architecture.

Conceptually an adapted projection computes:

```text
y = W x + (alpha / rank) B(A x)
```

The existing matrix W stays frozen. A and B are new trainable matrices. PEFT's default initialization makes B zero and A random, so the initial adapter contribution is zero. On the first backward pass, A can have zero gradient while B receives a nonzero gradient; that is expected.

`lora_dropout=0.0` makes this benchmark simpler by introducing no adapter dropout. `bias="none"` keeps existing biases frozen.

There are 24 layers × 2 target projections × 2 adapter matrices = 96 adapter tensors. A tensor contains many numerical parameters; 96 tensors is not 96 trainable numbers.

`requires_grad` tells PyTorch whether a parameter needs gradients. We collect only parameters with this flag enabled, verify their names identify LoRA parameters, and snapshot their starting values.

## The optimizer: lines 131–133

```python
optimizer = torch.optim.AdamW(
    list(trainable.values()), lr=1e-4, weight_decay=0.0
)
```

Only adapter parameters are passed to the optimizer. AdamW keeps running statistics of gradients and uses them to calculate adaptive updates. Weight decay is explicitly zero for this initial benchmark. LoRA limits which parameters are trained; SFT defines the learning task and objective.

## The core loop: lines 135–167

```python
optimizer.zero_grad(set_to_none=True)
output = model(**batch)
loss = output.loss
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(
    list(trainable.values()), 1.0, error_if_nonfinite=True
)
optimizer.step()
```

### Clear gradients

PyTorch accumulates gradients by default. `zero_grad` clears leftovers from the previous step; `set_to_none=True` releases those gradient buffers instead of filling them with zeros. Our script intentionally does no gradient accumulation.

### Forward pass and loss

`**batch` expands the dictionary into named arguments: `input_ids`, `attention_mask`, and `labels`. This invokes the model's forward computation, not its `generate` method.

Qwen produces vocabulary scores called logits for each sequence position. Because labels are supplied, Transformers also calculates next-token cross-entropy loss, ignoring target labels equal to -100.

The model shifts the targets internally. We do not shift `labels` a second time. The prediction after the assistant header is scored against the first SQL token, and the prediction after `SELECT` is scored against the next answer token.

For one target token, loss is `-log(probability assigned to the correct token)`. Higher correct-token probability gives lower loss. The returned scalar averages the scored positions for this example. Token IDs are categorical labels: loss is not the numerical distance between two token IDs.

Teacher forcing allows all positions in the known sequence to be processed together under causal attention. This helps explain why a training step can be quicker than generating a whole answer, which needs successive forward calls.

### Backward pass

`loss.backward()` calculates derivatives: how a small parameter change would affect the loss. Gradients are stored in the trainable parameters' `.grad` fields. This line does not update weights.

Frozen layers still participate in the forward computation and in carrying gradients to earlier adapters. Freezing base weights saves their gradient/optimizer storage, but does not remove the base model's computations.

### Clip gradients

`clip_grad_norm_` measures the combined gradient norm across the adapters. If it exceeds 1, it scales the gradients down together. The returned value is the norm before clipping. We check for finite and nonzero gradients and ensure frozen parameters did not acquire gradients.

### Update

`optimizer.step()` changes adapter parameters using those gradients and AdamW's running statistics. This is the actual learning update.

We log the loss before this update, step duration, gradient norm, and memory. Consecutive steps usually use different examples, so their losses need not decrease monotonically. The same-example probe is the cleaner before/after comparison, but remains a training-data check.

## Verification and saving: lines 169–181

After the loop we clear gradients and measure the probe again. We compare all frozen weights with their initial fingerprint and each adapter tensor with its initial values. In our run all 96 adapter tensors changed and all frozen weights stayed unchanged.

`model.save_pretrained` is PEFT's save method here: it saves adapters and their configuration, not another copy of Qwen. We explicitly avoid saving embedding layers. The adapter weights file is about 2.18 MB, compared with roughly 1 GB for the downloaded base snapshot.

We read the saved safetensors file back and compare every tensor to the live adapter state. This verifies serialization, not yet a fresh end-to-end model reload. The adapter still requires the original base model for inference.

The saved benchmark adapter is not an exact-resume training checkpoint: optimizer state is not saved. Full version 1 training will start fresh.

## Timing and the report: lines 183–229

We exclude the first step from the timing sample and use the median of the remaining seven. With batch size one and no accumulation, one full epoch would have 196 updates. Multiplying the median by 196 gives a rough training-only estimate.

`benchmark.json` records the settings, data/script hashes, package versions, selected examples, losses, timings, memory, and verification results. `steps.jsonl` is flushed after each step so completed measurements survive an interruption.

This benchmark does not select a checkpoint, evaluate validation/test accuracy, or show that the remaining pilot SQL mistake has been fixed. Those are later experimental steps.

## Questions to test your understanding

1. Which line changes parameter values, and which line only calculates gradients?
2. Why can prompt tokens affect learning even though their target labels are -100?
3. Why does `model.train()` not make frozen base parameters trainable?
4. Why can lower loss on the probe coexist with unchanged or worse held-out accuracy?
5. Why is an adapter file insufficient without the base model?

References: [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora) and [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW).
