"""Area D - make the Transformers smaller and faster with dynamic INT8 quantization.

The weights of all Linear layers are stored as 8-bit integers (INT8) instead of
32-bit numbers (FP32). The word embeddings stay FP32.
We quantize the distilled student (used in the app) and, for comparison, the teacher.
It saves models/student_int8.pt and models/teacher_int8.pt.

How to run:  python -m src.quantize
"""
import torch
from transformers import AutoModelForSequenceClassification

from .config import MODELS_DIR
from .models import model_size_mb, quantize_bert


def main():
    for name in ["student", "teacher"]:
        fp32 = AutoModelForSequenceClassification.from_pretrained(str(MODELS_DIR / f"{name}_fp32")).eval()
        int8 = quantize_bert(fp32)
        torch.save(int8.state_dict(), MODELS_DIR / f"{name}_int8.pt")
        print(f"{name}: FP32 {model_size_mb(name + '_fp32'):.1f} MB -> INT8 {model_size_mb(name + '_int8'):.1f} MB")


if __name__ == "__main__":
    main()
