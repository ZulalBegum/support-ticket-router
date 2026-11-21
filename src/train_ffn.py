"""Area A - our own fastText word vectors + a feedforward neural network.

- fastText is trained only on the training messages. It also learns from parts of
  words (character n-grams), so it can give a vector even to a misspelled word.
- A message = the average of its word vectors (100 numbers).
- Network: 100 -> 256 -> 151, ReLU and dropout 0.3, softmax output.
- Trained with SGD and backpropagation.

How to run:  python -m src.train_ffn
"""
import numpy as np
import torch
import torch.nn as nn
from gensim.models import FastText
from sklearn.metrics import accuracy_score

from .config import MODELS_DIR, SEED
from .data import load_labels, load_split
from .preprocess import SpellCorrector

EMB_DIM = 100


class FFN(nn.Module):
    def __init__(self, n_labels: int, in_dim: int = EMB_DIM, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_labels))

    def forward(self, x):
        return self.net(x)


def embed(texts, ft: FastText) -> np.ndarray:
    """Average word vector of each message (cleaned text, words split at spaces)."""
    out = np.zeros((len(texts), ft.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.split()
        if words:
            out[i] = np.mean([ft.wv[w] for w in words], axis=0)
    return out


def main(epochs: int = 60, lr: float = 0.05, batch_size: int = 32):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, val = load_split("train"), load_split("val")
    labels = load_labels()
    lab2id = {l: i for i, l in enumerate(labels)}
    speller = SpellCorrector.load(MODELS_DIR / "speller.json")
    tr_text = [speller.correct(t)[0] for t in train.text]
    va_text = [speller.correct(t)[0] for t in val.text]

    # 1) train fastText (skip-gram with character n-grams) on the training messages only
    ft = FastText([t.split() for t in tr_text], vector_size=EMB_DIM, window=5, min_count=1,
                  sg=1, epochs=20, min_n=3, max_n=5, bucket=50000, workers=4, seed=SEED)
    ft.save(str(MODELS_DIR / "fasttext.model"))

    # 2) train the feedforward network with SGD + backpropagation
    Xtr = torch.tensor(embed(tr_text, ft))
    ytr = torch.tensor([lab2id[l] for l in train.intent])
    Xva = torch.tensor(embed(va_text, ft))
    yva = np.array([lab2id[l] for l in val.intent])
    model = FFN(len(labels))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()          # backpropagation
            opt.step()               # SGD update
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = accuracy_score(yva, model(Xva).argmax(1).numpy())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3} | train loss {total / len(Xtr):.4f} | val accuracy {acc:.4f}")
    torch.save(best_state, MODELS_DIR / "ffn.pt")
    print(f"best validation accuracy = {best_acc:.4f}  -> saved models/ffn.pt")


if __name__ == "__main__":
    main()
