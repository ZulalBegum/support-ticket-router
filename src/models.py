"""Load the models. Every model has the same function:
predict_proba(clean_text) -> probabilities for the 151 labels (same order as models/labels.json).

  tfidf         - tf-idf + logistic regression (area B, classical baseline)
  ffn           - our fastText vectors + feedforward network (area A)
  teacher_fp32  - fine-tuned BERT-base, normal 32-bit (area C)
  teacher_int8  - BERT-base after INT8 quantization (area D, for comparison)
  student_fp32  - small BERT trained by distillation from the teacher (area D)
  student_int8  - distilled small BERT after INT8 quantization (area D) -> used in the app
"""
import platform

import joblib
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_LEN, MODELS_DIR
from .data import load_labels

MODEL_NAMES = {
    "student_int8": "BERT-small INT8 (distilled + quantized)",
    "student_fp32": "BERT-small FP32 (distilled)",
    "teacher_int8": "BERT-base INT8",
    "teacher_fp32": "BERT-base FP32 (teacher)",
    "ffn": "fastText + FFN",
    "tfidf": "tf-idf + LogReg",
}

MODEL_FILES = {  # files of each model (used to measure the size in MB)
    "tfidf": ["tfidf_lr.joblib"],
    "ffn": ["fasttext.model", "fasttext.model.wv.vectors_ngrams.npy", "fasttext.model.wv.vectors.npy", "ffn.pt"],
    "teacher_fp32": ["teacher_fp32/model.safetensors"],
    "teacher_int8": ["teacher_int8.pt"],
    "student_fp32": ["student_fp32/model.safetensors"],
    "student_int8": ["student_int8.pt"],
}


def set_quant_engine():
    """Choose the quantization engine for this CPU: qnnpack for Apple M1/M2 (ARM), x86 for Intel/AMD."""
    engines = torch.backends.quantized.supported_engines
    if platform.machine().lower() in ("arm64", "aarch64") and "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "x86" in engines:
        torch.backends.quantized.engine = "x86"
    elif "fbgemm" in engines:
        torch.backends.quantized.engine = "fbgemm"


def quantize_bert(model):
    set_quant_engine()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def model_size_mb(key: str) -> float:
    return sum((MODELS_DIR / f).stat().st_size for f in MODEL_FILES[key] if (MODELS_DIR / f).exists()) / 1e6


class TfidfModel:
    def __init__(self):
        self.pipe = joblib.load(MODELS_DIR / "tfidf_lr.joblib")
        order = list(self.pipe.classes_)
        self.idx = [order.index(l) for l in load_labels()]   # put columns in our label order

    def predict_proba(self, text):
        return self.pipe.predict_proba([text])[0][self.idx]


class FFNModel:
    def __init__(self):
        from gensim.models import FastText
        from .train_ffn import FFN
        self.ft = FastText.load(str(MODELS_DIR / "fasttext.model"))
        self.net = FFN(len(load_labels()))
        self.net.load_state_dict(torch.load(MODELS_DIR / "ffn.pt"))
        self.net.eval()

    def predict_proba(self, text):
        from .train_ffn import embed
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(embed([text], self.ft))), dim=1)[0].numpy()


class BertModel:
    def __init__(self, name: str, quantized: bool):
        path = str(MODELS_DIR / f"{name}_fp32")
        self.tok = AutoTokenizer.from_pretrained(path)
        if quantized:
            # make an empty model, quantize it, then load the saved INT8 weights
            model = quantize_bert(AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(path)))
            model.load_state_dict(torch.load(MODELS_DIR / f"{name}_int8.pt", weights_only=False))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model = model.eval()

    def predict_proba(self, text):
        enc = self.tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.inference_mode():
            return torch.softmax(self.model(**enc).logits, dim=1)[0].numpy()


def load_model(key: str):
    if key == "tfidf":
        return TfidfModel()
    if key == "ffn":
        return FFNModel()
    name, prec = key.split("_")
    return BertModel(name, quantized=(prec == "int8"))


def predict_label(probs: np.ndarray, labels: list, threshold: float):
    """Best label and its confidence. Below the threshold -> out-of-scope."""
    i = int(np.argmax(probs))
    conf = float(probs[i])
    label = labels[i] if conf >= threshold else "oos"
    return label, conf, labels[i]
