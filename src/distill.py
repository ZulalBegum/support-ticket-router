"""Area D - knowledge distillation: teach a SMALL BERT (student) to copy the BIG BERT (teacher).

- Teacher: our fine-tuned bert-base (12 layers, ~110M parameters), not changed.
- Student: google/bert_uncased_L-4_H-512_A-8 (4 layers, ~29M parameters).
- Loss = 0.5 * normal cross-entropy with the true labels
       + 0.5 * KL divergence between student and teacher "soft" probabilities (temperature T = 2).
  The soft probabilities show the student which wrong labels are "almost right".

Run on a GPU (Colab T4). Saves models/student_fp32/.
How to run:  python -m src.distill --epochs 8
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from .config import MODELS_DIR, SEED, STUDENT_BASE
from .data import load_labels
from .train_bert import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=STUDENT_BASE)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = load_labels()
    lab2id = {l: i for i, l in enumerate(labels)}
    from .preprocess import SpellCorrector
    speller = SpellCorrector.load(MODELS_DIR / "speller.json")

    teacher_dir = str(MODELS_DIR / "teacher_fp32")
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir).to(device).eval()
    tok = AutoTokenizer.from_pretrained(teacher_dir)   # both models use the same uncased vocabulary
    student = AutoModelForSequenceClassification.from_pretrained(
        args.student, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=lab2id, ignore_mismatched_sizes=True).to(device)

    collate = DataCollatorWithPadding(tok)
    keep = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    train_ds = make_dataset("train", tok, speller, lab2id)
    val_ds = make_dataset("val", tok, speller, lab2id)
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=128, collate_fn=collate)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    T, a = args.temperature, args.alpha
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        run_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.no_grad():
                t_logits = teacher(**batch).logits
            s_logits = student(**batch).logits
            hard = F.cross_entropy(s_logits, y)
            soft = F.kl_div(F.log_softmax(s_logits / T, dim=-1), F.softmax(t_logits / T, dim=-1),
                            reduction="batchmean") * (T * T)
            loss = a * hard + (1 - a) * soft
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
        # validation accuracy
        student.eval()
        preds, gold = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                gold += batch.pop("labels").tolist()
                preds += student(**batch).logits.argmax(-1).tolist()
        acc = accuracy_score(gold, preds)
        print(f"epoch {epoch} | loss {run_loss / len(train_dl):.4f} | val accuracy {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            student.save_pretrained(str(MODELS_DIR / "student_fp32"))
            tok.save_pretrained(str(MODELS_DIR / "student_fp32"))
    print(f"best validation accuracy = {best_acc:.4f} -> saved models/student_fp32")


if __name__ == "__main__":
    main()
