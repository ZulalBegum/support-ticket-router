"""Settings used by all files: folders, model names, teams and limits."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# The trained models are saved on Hugging Face, not on GitHub (they are too big).
HF_REPO = os.environ.get("HF_REPO", "zulalbgm22/support-ticket-router")

# CLINC150 dataset (Larson et al., 2019), original files on GitHub
DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
DOMAINS_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/domains.json"

# Teacher = big Transformer, student = small Transformer (knowledge distillation)
TEACHER_BASE = os.environ.get("TEACHER_BASE", "bert-base-uncased")
STUDENT_BASE = os.environ.get("STUDENT_BASE", "google/bert_uncased_L-4_H-512_A-8")

OOS = "oos"              # label of out-of-scope messages
MAX_LEN = 32             # max number of tokens per message for BERT
SEED = 42

# Each CLINC150 domain is one support team
TEAMS = {
    "banking": "Banking team",
    "credit_cards": "Credit card team",
    "kitchen_and_dining": "Kitchen & dining team",
    "home": "Home & orders team",
    "auto_and_commute": "Auto & commute team",
    "travel": "Travel team",
    "utility": "Utility team",
    "work": "Work & HR team",
    "small_talk": "Chat bot (small talk)",
    "meta": "Assistant settings team",
}
HUMAN_AGENT = "Human agent"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
