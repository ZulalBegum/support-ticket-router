"""Upload and download the trained models on Hugging Face (they are too big for GitHub).

Upload after training:   python -m src.hub upload    (you must be logged in to Hugging Face)
Download:                python -m src.hub download  (app.py does this automatically)
"""
import sys

from huggingface_hub import HfApi, snapshot_download

from .config import HF_REPO, MODELS_DIR

IGNORE = ["checkpoints/*", "*.ipynb_checkpoints*"]
NEEDED = ["labels.json", "intent_domain.json", "speller.json", "thresholds.json", "tfidf_lr.joblib",
          "fasttext.model", "ffn.pt", "teacher_fp32/config.json", "student_fp32/config.json",
          "teacher_int8.pt", "student_int8.pt"]


def upload():
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(MODELS_DIR), repo_id=HF_REPO, repo_type="model",
                      ignore_patterns=IGNORE)
    print(f"uploaded models/ to https://huggingface.co/{HF_REPO}")


def download(force: bool = False):
    if not force and all((MODELS_DIR / f).exists() for f in NEEDED):
        return
    print(f"downloading models from https://huggingface.co/{HF_REPO} ...")
    snapshot_download(repo_id=HF_REPO, repo_type="model", local_dir=str(MODELS_DIR),
                      ignore_patterns=IGNORE)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    upload() if cmd == "upload" else download(force=True)
