# Training modes and command-line arguments

There are three different activities; "pilot" and "benchmark" are not interchangeable:

| Activity | Command after the Python executable | Data | Updates weights? |
|---|---|---|---|
| Pilot evaluation | `evaluate.py --split pilot` | 42 pilot questions | No |
| Short training benchmark | `train.py` | Seven selected training examples, eight updates | Yes, temporary fresh adapters |
| Full training | `train.py --full --epochs 3` | All 196 training examples, three passes | Yes, fresh adapters |

## Why eight updates?

The benchmark chooses the longest tokenized example from each of the seven query families, without selecting by model loss. It first performs one warm-up update, then processes the seven selected examples. That means eight real updates but seven unique examples: the first chosen example appears twice.

Every update runs forward, loss, backward, gradient clipping, and the optimizer step. The warm-up changes adapter weights too; it is only excluded from the timing summary because first-use overhead can distort speed estimates. The benchmark also checks memory, gradients, unchanged frozen weights, and whether saved adapter tensors match.

This is a quick engineering check, not a complete experiment or a quality score. Full training does not continue from those eight updates: it starts from the original base model and fresh adapters.

## How arguments select the mode

Near the bottom of `train.py`, the existing code is:

```python
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
```

`argparse` is part of Python's standard library. It reads command-line arguments and turns them into named values:

- `ArgumentParser` creates the reader and its help description.
- `--full` uses `store_true`: absent means `False`, present means `True`.
- `--epochs` takes an integer; if omitted its value is 3.
- `parse_args()` reads what you typed after the script name into `args`.
- The positive-number check rejects zero or negative epoch counts.
- The final `if` chooses which function to call, passing the epoch count to full training.

For `train.py --full --epochs 3`, think: `args.full = True`, `args.epochs = 3`, therefore call `full_main(3)`. These arguments do not edit the file: they configure this particular execution. `train.py --epochs 5` still runs the benchmark because `--full` is absent. `train.py --help` prints the options and exits without training.
