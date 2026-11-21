"""Area B - ELIZA-style rules for greetings and small talk.

Each rule is a regular expression and a reply. Like ELIZA, some replies reuse
words from the message after "reflecting" them (I -> you, my -> your).
These rules are used when the message is a greeting / small talk,
so the bot answers and no human team is needed.
"""
import random
import re

REFLECT = {"i": "you", "me": "you", "my": "your", "am": "are", "i'm": "you're",
           "mine": "yours", "you": "I", "your": "my", "are": "am"}

RULES = [
    (r"^(hi|hello|hey|good (morning|afternoon|evening))\b", [
        "Hello! How can I help you today?", "Hi there! What can I do for you?"]),
    (r"\b(thanks|thank you|thx)\b", [
        "You're welcome! Anything else I can help with?"]),
    (r"\b(bye|goodbye|see you)\b", [
        "Goodbye! Have a nice day."]),
    (r"\bhow are you\b", [
        "I'm fine, thank you. How can I help you?"]),
    (r"\bare you (a )?(bot|robot|human|real)\b", [
        "I'm a virtual assistant. I can send your message to the right team."]),
    (r"\bi feel (.+)", [
        "Why do you feel {0}?", "I'm sorry you feel {0}. Tell me what happened."]),
    (r"\bi need (.+)", [
        "Why do you need {0}? Tell me a bit more so I can find the right team."]),
]
# rules 0-4 are clear small talk: they can answer even before the models run
STANDALONE_RULES = 5
FALLBACK = ["Tell me more, please.", "How can I help you with that?"]
COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in RULES]


def reflect(fragment: str) -> str:
    """Change the point of view: 'my order' -> 'your order'."""
    return " ".join(REFLECT.get(w.lower(), w) for w in fragment.split())


def rule_match(text: str, standalone_only: bool = False):
    """Return (pattern, reply) if a rule matches, else None."""
    t = text.strip().rstrip("?!.")
    rules = COMPILED[:STANDALONE_RULES] if standalone_only else COMPILED
    for pattern, replies in rules:
        m = pattern.search(t)
        if m:
            groups = [reflect(g) for g in m.groups() if g] if pattern.groups else []
            reply = random.choice(replies)
            try:
                reply = reply.format(*groups)
            except IndexError:
                pass
            return pattern.pattern, reply
    return None


def reply(text: str) -> str:
    m = rule_match(text)
    return m[1] if m else random.choice(FALLBACK)


if __name__ == "__main__":
    for s in ["hello!", "thank you so much", "I feel tired of my old phone", "are you a robot?",
              "what is my balance"]:
        print(f"{s!r:35} -> {rule_match(s)}")
