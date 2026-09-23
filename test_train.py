"""Check training masks and the actual next-token loss without loading Qwen weights."""

import json
import unittest
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from transformers.loss.loss_utils import ForCausalLMLoss

from train import MODEL_DIR, TRAIN_PATH, PROJECT_ROOT, make_batch, validation_loss


class TrainingInputTests(unittest.TestCase):
    def test_every_training_example_scores_only_answer_and_ending(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        paths = [TRAIN_PATH, PROJECT_ROOT / "data" / "validation.jsonl"]
        records = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            with self.subTest(example=record["id"]):
                batch = make_batch(tokenizer, record)
                labels = batch["labels"][0]
                mask = labels != -100
                answer_ids = batch["input_ids"][0][mask]
                self.assertEqual(tokenizer.decode(answer_ids, skip_special_tokens=False),
                                 record["sql"] + "<|im_end|>\n")
                self.assertTrue(torch.equal(labels[mask], answer_ids))
                self.assertEqual(labels[0].item(), -100)
                self.assertEqual(batch["input_ids"].shape, batch["attention_mask"].shape)
                self.assertTrue((batch["attention_mask"] == 1).all().item())

    def test_loss_shift_and_mask_against_manual_cross_entropy(self):
        # Target at position 2 must be predicted by logits at position 1.
        logits = torch.tensor([[[2., 0., -1.], [0., 2., -1.], [-1., 0., 2.], [1., 1., 1.]]],
                              requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2]])
        actual = ForCausalLMLoss(logits, labels, vocab_size=3)
        expected = -(logits[0, 1].log_softmax(-1)[1] + logits[0, 2].log_softmax(-1)[2]) / 2
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 3].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 1:3].abs().sum().item(), 0)

    def test_validation_weights_tokens_and_does_not_train(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, labels, example_loss):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError("Validation must disable training mode and gradients")
                return SimpleNamespace(loss=self.weight * example_loss)

        model = FakeModel()
        examples = [
            ({"id": "short"}, {"labels": torch.tensor([[-100, 1]]), "example_loss": 2.0}),
            ({"id": "long"}, {"labels": torch.tensor([[-100, 1, 2, 3]]), "example_loss": 4.0}),
        ]
        result = validation_loss(model, examples)
        self.assertEqual(result["loss"], 3.5)  # (2*1 + 4*3) / 4, not (2+4)/2.
        self.assertEqual(result["scored_tokens"], 4)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(model.weight.item(), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
