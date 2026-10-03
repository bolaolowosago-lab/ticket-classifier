"""Fine-tune DistilBERT on the same split and compare against the baseline.
A GPU helps a lot. Free option: run this in Google Colab with a T4 GPU."""
import argparse
from pathlib import Path

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from data import load_split
from evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, y=None):
        self.enc, self.y = enc, y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.y is not None:
            item["labels"] = torch.tensor(self.y[i])
        return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args()

    train, test = load_split(args.data, args.text_col, args.label_col)
    labels = sorted(train[args.label_col].unique())
    l2i = {l: i for i, l in enumerate(labels)}

    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels),
        id2label={i: l for l, i in l2i.items()}, label2id=l2i,
    )

    def encode(texts):
        return tok(list(texts), truncation=True, padding="max_length",
                   max_length=args.max_len)

    train_ds = DS(encode(train[args.text_col]), train[args.label_col].map(l2i).tolist())
    test_ds = DS(encode(test[args.text_col]))

    targs = TrainingArguments(
        output_dir=str(ROOT / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
        learning_rate=5e-5, weight_decay=0.01, save_strategy="no",
        logging_steps=50, report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds)
    trainer.train()

    logits = trainer.predict(test_ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred = [labels[i] for i in probs.argmax(axis=1)]

    evaluate(test[args.text_col], test[args.label_col], pred, probs.max(axis=1),
             labels, "distilbert")

    model.save_pretrained(ROOT / "models" / "distilbert")
    tok.save_pretrained(ROOT / "models" / "distilbert")
    print("saved models/distilbert")


if __name__ == "__main__":
    main()
