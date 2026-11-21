"""Customer support ticket router (Gradio app).

- You type a customer message.
- It finds product / order codes and fixes spelling mistakes.
- Greetings and small talk get a rule-based (ELIZA-style) reply.
- The model predicts the intent with a confidence score and shows the team.
- Low confidence or out-of-scope messages go to a human agent.
- A table shows the predictions of all models side by side.
- The default model is the distilled + quantized BERT-small INT8 (requirement R5).

How to run:  python app.py   then open http://127.0.0.1:7860
"""
import gradio as gr
import pandas as pd

from src.config import HUMAN_AGENT
from src.hub import download
from src.models import MODEL_NAMES
from src.router import Router

download()  # download the models from Hugging Face the first time
print("loading models ...")
ROUTER = Router(list(MODEL_NAMES))
NAME_TO_KEY = {v: k for k, v in MODEL_NAMES.items()}
DEFAULT = MODEL_NAMES["student_int8"]

EXAMPLES = [
    "my card got declined at the store, can you check why?",
    "I want to book a flight to Rome next friday",
    "what is the acount balanse on my checking account",
    "how long should I cook chicken in the oven",
    "hello!",
    "who won the football world cup in 1998",
]


def run(message, model_name):
    message = (message or "").strip()
    if not message:
        return "", "", pd.DataFrame()
    res = ROUTER.route(message, NAME_TO_KEY[model_name])

    icon = "🧑‍💼" if res["team"] == HUMAN_AGENT else ("🤖" if res["reply"] else "✅")
    card = f"## {icon} Routed to: **{res['team']}**\n\n"
    if res["by_rule"]:
        card += "Matched a small-talk rule (ELIZA-style), no team needed.\n\n"
    elif res["intent"] == "oos":
        why = ("predicted out-of-scope" if res["top_intent"] == "oos"
               else f"low confidence ({res['confidence']:.0%}) for `{res['top_intent']}`")
        card += f"Sent to a human agent: {why}.\n\n"
    else:
        card += f"Intent: `{res['intent']}` · confidence **{res['confidence']:.0%}**\n\n"
    if res["reply"]:
        card += f"> 🤖 **Bot reply:** {res['reply']}\n\n"
    card += f"Model: {model_name} · {res['latency_ms']:.1f} ms"

    info = (f"**Normalized text:** {res['clean'] or '-'}\n\n"
            f"**Spelling fixes:** {', '.join(f'{a} → {b}' for a, b in res['changes']) or 'none'}\n\n"
            f"**Product / order codes:** {', '.join(res['codes']) or 'none'}")

    rows = []
    for key, name in MODEL_NAMES.items():
        r = ROUTER.predict(message, key)
        rows.append([name, r["intent"], round(r["confidence"], 3), r["team"], round(r["latency_ms"], 2)])
    table = pd.DataFrame(rows, columns=["Model", "Intent", "Confidence", "Team", "Latency (ms)"])
    return card, info, table


with gr.Blocks(title="Customer Support Ticket Router") as demo:
    gr.Markdown("# 📨 Customer Support Ticket Router\n"
                "Type a customer message. It is routed to the right team instantly; "
                "unclear or out-of-scope messages go to a human agent.")
    with gr.Row():
        with gr.Column(scale=1):
            msg = gr.Textbox(label="Customer message", lines=3, placeholder="e.g. my card was declined ...")
            model_dd = gr.Dropdown(list(MODEL_NAMES.values()), value=DEFAULT, label="Routing model")
            btn = gr.Button("Route message", variant="primary")
            gr.Examples(EXAMPLES, inputs=msg)
        with gr.Column(scale=1):
            card = gr.Markdown()
            info = gr.Markdown()
    table = gr.Dataframe(label="All models side by side", interactive=False)
    btn.click(run, [msg, model_dd], [card, info, table])
    msg.submit(run, [msg, model_dd], [card, info, table])

if __name__ == "__main__":
    demo.launch()
