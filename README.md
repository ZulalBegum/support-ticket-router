# Customer Support Ticket Router — SWE015

A support app that reads a free-text customer message, cleans it, predicts the intent with a
confidence score and routes it to the right team. Low-confidence or out-of-scope messages go to a
**human agent**; greetings and small talk get a rule-based (ELIZA-style) reply. The predictions of
all models are shown side by side. The app runs the **distilled + quantized BERT-small (INT8)**.

| Area | Component | File |
|---|---|---|
| B · Classical NLP | regex normalization + product/order code extraction | `src/preprocess.py` |
| B · Classical NLP | spelling correction by minimum edit distance | `src/preprocess.py` |
| B · Classical NLP | ELIZA-style rules for small talk | `src/smalltalk.py` |
| B · Classical NLP | tf-idf + logistic regression baseline | `src/train_baselines.py` |
| A · Neural networks | own fastText vectors (averaged) + feedforward net, SGD + backprop | `src/train_ffn.py` |
| C · Transformers | BERT-base fine-tuned for intent classification (teacher) | `src/train_bert.py` |
| D · Efficient models | knowledge distillation into BERT-small, then INT8 quantization | `src/distill.py`, `src/quantize.py` |

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/ZulalBegum/support-ticket-router.git
cd support-ticket-router
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run (single command)

```bash
python app.py
```

Open http://127.0.0.1:7860. On the first start the trained models are downloaded automatically
from the Hugging Face Hub: <https://huggingface.co/zulalbgm22/support-ticket-router>.

## Reproduce training and results

1. Train on Google Colab (GPU): open `notebooks/train_colab.ipynb` and run all cells.
2. Benchmark on your CPU (same test set for all models):

```bash
python -m src.prepare_data
python -m src.benchmark --runs 200
```

The results are written to `results/results.md`.

## Data

CLINC150: Larson et al. (2019), *An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction*, EMNLP. Data: <https://github.com/clinc/oos-eval> (`data_full.json`).
150 intents in 10 domains + out-of-scope. Each domain is one support team. The official test
split (4,500 in-scope + 1,000 out-of-scope) is the held-out test set for all models.
