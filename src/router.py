"""The full routing pipeline used by the app.

message -> find codes -> clean + spelling correction -> small talk rules?
        -> model predicts the intent -> low confidence or out-of-scope? -> team
"""
import json
import time

from .config import HUMAN_AGENT, MODELS_DIR, OOS, TEAMS
from .data import load_intent_domain, load_labels
from .models import load_model, predict_label
from .preprocess import SpellCorrector, extract_codes
from . import smalltalk


class Router:
    def __init__(self, model_keys):
        self.labels = load_labels()
        self.domain = load_intent_domain()
        self.speller = SpellCorrector.load(MODELS_DIR / "speller.json")
        self.thresholds = json.loads((MODELS_DIR / "thresholds.json").read_text(encoding="utf-8"))
        self.models = {k: load_model(k) for k in model_keys}

    def predict(self, text: str, key: str) -> dict:
        """Run one model on a message (no small talk rules). Time includes cleaning."""
        t0 = time.perf_counter()
        clean, changes = self.speller.correct(text)
        probs = self.models[key].predict_proba(clean)
        intent, conf, top = predict_label(probs, self.labels, self.thresholds.get(key, 0.0))
        ms = (time.perf_counter() - t0) * 1000
        domain = self.domain.get(intent, OOS)
        team = HUMAN_AGENT if intent == OOS else TEAMS[domain]
        return {"intent": intent, "top_intent": top, "confidence": conf, "domain": domain,
                "team": team, "latency_ms": ms, "clean": clean, "changes": changes}

    def route(self, text: str, key: str) -> dict:
        """Full pipeline for the app."""
        codes = extract_codes(text)
        rule = smalltalk.rule_match(text, standalone_only=True) if len(text.split()) <= 5 else None
        res = self.predict(text, key)
        res["codes"] = codes
        res["reply"] = None
        res["by_rule"] = False
        if rule:                                   # clear greeting / thanks / bye
            res.update(team=TEAMS["small_talk"], reply=rule[1], by_rule=True)
        elif res["domain"] == "small_talk":        # the model says it is small talk
            res["reply"] = smalltalk.reply(text)
        return res
