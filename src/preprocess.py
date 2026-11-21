"""Area B - clean the customer message before the models see it.

1. Regex: find product / order codes like "DW-6534" or "#48213" and save them.
2. Regex normalization: lower case, remove links and extra symbols.
3. Spelling correction with minimum edit distance: a word that is not in our
   word list (built from the training messages) is replaced by the closest
   known word, e.g. "acount" -> "account", "dishwaser" -> "dishwasher".
"""
import json
import re
from collections import Counter

# product / model / order codes: DW-6534, AB1234, #48213, order 55120
CODE_RE = re.compile(r"\b[A-Za-z]{1,4}-?\d{3,6}\b|#\d{3,8}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_WORD_RE = re.compile(r"[^a-z0-9' ]+")
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"^[a-z]+$")

# product names of the company, so they are also corrected when misspelled
PRODUCT_WORDS = ["dishwasher", "refrigerator", "fridge", "freezer", "oven", "microwave",
                 "washer", "dryer", "thermostat", "vacuum", "blender", "toaster"]


def extract_codes(text: str) -> list[str]:
    """Return all product / order codes found in the message."""
    return [c.upper() for c in CODE_RE.findall(text)]


def normalize(text: str) -> str:
    """Lower case, remove links, codes and symbols, keep words and numbers."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = CODE_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance: the smallest number of insert, delete or change steps to turn a into b."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _deletes(word: str, depth: int) -> set[str]:
    """All strings we get by deleting up to `depth` letters (used to find candidates fast)."""
    out, frontier = {word}, {word}
    for _ in range(depth):
        frontier = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        out |= frontier
    return out


class SpellCorrector:
    """Correct unknown words to the closest known word (edit distance 1, or 2 for long words)."""

    def __init__(self, counts: dict):
        self.counts = counts
        self.index = {}
        for w in counts:
            for d in _deletes(w, 2 if len(w) >= 8 else 1):
                self.index.setdefault(d, []).append(w)

    @classmethod
    def from_texts(cls, texts, min_count: int = 2):
        c = Counter(w for t in texts for w in normalize(t).split() if WORD_RE.match(w))
        counts = {w: n for w, n in c.items() if n >= min_count and len(w) >= 2}
        for w in PRODUCT_WORDS:
            counts.setdefault(w, min_count)
        return cls(counts)

    def correct_word(self, w: str) -> str:
        if w in self.counts or len(w) < 4 or not WORD_RE.match(w):
            return w
        max_d = 2 if len(w) >= 8 else 1
        cands = {c for d in _deletes(w, max_d) for c in self.index.get(d, [])}
        best = None
        for c in cands:
            d = edit_distance(w, c)
            if d <= max_d:
                key = (d, -self.counts[c])          # closest first, then most frequent
                if best is None or key < best[0]:
                    best = (key, c)
        return best[1] if best else w

    def correct(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Return the corrected text and the list of (wrong word, correction)."""
        words, changes = [], []
        for w in normalize(text).split():
            c = self.correct_word(w)
            if c != w:
                changes.append((w, c))
            words.append(c)
        return " ".join(words), changes

    def save(self, path):
        path.write_text(json.dumps(self.counts), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    sc = SpellCorrector.from_texts(["my account balance please", "the dishwasher is broken",
                                    "check my account", "what is my balance"])
    for s in ["My acount balanse? ref DW-6534", "dishwaser not draining #48213"]:
        print(s, "->", sc.correct(s), extract_codes(s))
