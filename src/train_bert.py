"""Area C - fine-tune BERT (bert-base-uncased) for intent classification. This is the TEACHER.

A classification head (one Linear layer) is added on top of BERT and trained on 151 labels.
Run this on a GPU (Google Colab: Runtime > Change runtime type > T4 GPU).
The trained model is saved to models/teacher_fp32/.

How to run:  python -m src.train_bert --epochs 4
"""
import argparse
import inspect
import math

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from .config import MAX_LEN, MODELS_DIR, SEED, TEACHER_BASE
from .data import load_labels, load_split
from .preprocess import SpellCorrector


def make_dataset(name, tok, speller, lab2id):
    df = load_split(name)
    texts = [speller.correct(t)[0] for t in df.text]
    ds = Dataset.from_dict({"text": texts, "labels": [lab2id[l] for l in df.intent]})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=TEACHER_BASE)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", default="teacher_fp32")
    args = ap.parse_args()

    labels = load_labels()
    lab2id = {l: i for i, l in enumerate(labels)}
    speller = SpellCorrector.load(MODELS_DIR / "speller.json")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=lab2id, ignore_mismatched_sizes=True)
    train_ds = make_dataset("train", tok, speller, lab2id)
    val_ds = make_dataset("val", tok, speller, lab2id)

    def metrics(p):
        return {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}

    steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    targs_kwargs = dict(
        output_dir=str(MODELS_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=128,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=int(0.1 * steps),
        eval_strategy="epoch",
        evaluation_strategy="epoch",  # same setting, old name (for old versions)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )
    # use only the settings that this transformers version knows
    accepted = inspect.signature(TrainingArguments.__init__).parameters
    targs = TrainingArguments(**{k: v for k, v in targs_kwargs.items() if k in accepted})
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    print("validation:", trainer.evaluate())
    out = MODELS_DIR / args.out
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"saved model to {out}")


if __name__ == "__main__":
    main()
