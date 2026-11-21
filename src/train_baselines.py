"""Area B - classical baseline: tf-idf + logistic regression.

It also builds the word list for spelling correction (from the training messages only).
Input text = cleaned and spell-corrected message.

How to run:  python -m src.train_baselines
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import MODELS_DIR, SEED
from .data import load_split, make_labels
from .preprocess import SpellCorrector


def main():
    train, val = load_split("train"), load_split("val")
    make_labels(train)

    # word list for spelling correction
    speller = SpellCorrector.from_texts(train.text)
    speller.save(MODELS_DIR / "speller.json")
    print(f"spelling word list: {len(speller.counts)} words")

    clean = lambda texts: [speller.correct(t)[0] for t in texts]
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    pipe = Pipeline([("tfidf", features),
                     ("clf", LogisticRegression(C=10.0, max_iter=3000, random_state=SEED))])
    pipe.fit(clean(train.text), train.intent)
    joblib.dump(pipe, MODELS_DIR / "tfidf_lr.joblib", compress=3)
    acc = accuracy_score(val.intent, pipe.predict(clean(val.text)))
    print(f"tf-idf + LR | validation accuracy (all 151 labels) = {acc:.4f}")


if __name__ == "__main__":
    main()
