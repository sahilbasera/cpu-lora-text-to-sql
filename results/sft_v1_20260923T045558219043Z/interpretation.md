# Version 1 SFT result

Full CPU LoRA SFT completed from fresh adapters on the pinned Qwen2.5-0.5B-Instruct base. All 196 training examples were used once per epoch, verified from the logged IDs, for 588 updates across three epochs. Validation used 56 distinct held-out examples without parameter updates.

| Stage | Answer-token-weighted validation loss |
|---|---:|
| Before training | 0.18953457 |
| Epoch 1 | 0.00951156 |
| Epoch 2 | 0.00969625 |
| Epoch 3 | 0.01031525 |

The predeclared rule selected epoch 1 (lowest validation loss). Training loss continued falling after epoch 1, while validation loss rose slightly. This illustrates why we select on validation rather than automatically taking the final checkpoint. It is consistent with some overfitting, but a single small run does not establish the cause conclusively.

Selected adapter: `epoch_1/`. All three epoch checkpoints remain saved. Each saved adapter was checked against its live tensors. The selected adapter was also loaded from disk as a separate adapter on the unchanged base model and reproduced its validation score exactly in this run. Frozen base parameter fingerprints matched before and after training.

Training updates took about 9.9 minutes total; the measured run, including validation and verification, took 730.25 seconds (12.2 minutes), excluding Python imports before the timer began. Peak resident process memory was 3.16 GiB.

This is teacher-forced validation loss, not generated SQL execution accuracy. No final test examples were read or evaluated, and no post-training SQL accuracy result is available yet. The next experiment is a controlled base-versus-selected-adapter comparison using the same test questions and execution evaluator.
