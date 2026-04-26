"""
Verdict ⚖️ — Gradio Demo App
Integrates trained Qwen2.5-3B-GRPO model for live courtroom episodes.
Deploy this as app.py in your HuggingFace Space.
"""

import gradio as gr
import json
import random
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ── CONFIG ───────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
TRAINED_MODEL = "Imaginephoenix/verdict-qwen2.5-3b-grpo"
CASES_PATH   = "cases.json"
MAX_ROUNDS   = 3

# ── LOAD CASES ───────────────────────────────────────────────
with open(CASES_PATH) as f:
    CASES = json.load(f)

CASE_TITLES = [f"{c['id']} — {c['title']}" for c in CASES]
CASE_MAP    = {f"{c['id']} — {c['title']}": c for c in CASES}

# ── LOAD MODEL ───────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    TRAINED_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True,
)
model.eval()
print("✅ Model loaded!")

# ── PROMPTS ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert courtroom attorney in a legal simulation.
Make compelling, specific, and logically sound legal arguments.
Always reference specific evidence. Be concise (60-120 words).
Always respond in EXACTLY this format:
ACTION: [argue|counter|present_evidence|object|close]
ARGUMENT: [your argument here]"""

JUDGE_PROMPT = """You are an impartial judge evaluating a courtroom debate.
Review the full transcript carefully and decide who argued better.
Consider: logical consistency, evidence usage, specificity, and persuasiveness.
Respond in EXACTLY this format:
WINNER: [Prosecutor|Defense|Draw]
SCORE_PROSECUTOR: [0-10]
SCORE_DEFENSE: [0-10]
REASONING: [2-3 sentences explaining your decision]"""

# ── REWARD FUNCTION ──────────────────────────────────────────
LEGAL_KEYWORDS = [
    "evidence", "proves", "demonstrates", "therefore", "objection",
    "contract", "violation", "witness", "testimony", "facts", "record",
    "document", "liability", "negligence", "breach", "damages",
    "plaintiff", "defendant", "burden", "proof", "exhibit", "statute"
]

def score_argument(text: str) -> float:
    score = 0.0
    text_lower = text.lower()
    words = text_lower.split()

    has_action   = bool(re.search(r'action\s*:', text_lower))
    has_argument = bool(re.search(r'argument\s*:', text_lower))
    if has_action and has_argument:
        score += 0.25
    elif has_action or has_argument:
        score += 0.1
    else:
        score -= 0.2

    wc = len(words)
    if wc < 5:       score -= 0.3
    elif wc <= 60:   score += 0.15
    elif wc <= 150:  score += 0.25
    else:            score += 0.1

    kw = sum(1 for k in LEGAL_KEYWORDS if k in text_lower)
    score += min(kw * 0.05, 0.25)

    has_num  = bool(re.search(r'\b\d+\b', text))
    has_name = bool(re.search(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', text))
    score += (0.05 * has_num) + (0.05 * has_name)

    action_match = re.search(r'action\s*:\s*(\w+)', text_lower)
    if action_match:
        bonuses = {
            "present_evidence": 0.10,
            "counter": 0.08,
            "close": 0.08,
            "object": 0.05,
            "argue": 0.03
        }
        score += bonuses.get(action_match.group(1), 0.0)

    return round(min(max(score, -0.5), 1.0), 3)

# ── GENERATION ───────────────────────────────────────────────
def generate(messages, max_tokens=200, temperature=0.7):
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    return tokenizer.decode(
        output[0][input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()


def parse_response(text: str) -> dict:
    action = "argue"
    argument = text

    a = re.search(r'action\s*:\s*(\w+)', text, re.IGNORECASE)
    if a and a.group(1).lower() in ["argue","counter","present_evidence","object","close"]:
        action = a.group(1).lower()

    b = re.search(r'argument\s*:\s*(.+)', text, re.DOTALL | re.IGNORECASE)
    if b:
        argument = b.group(1).strip()

    return {"action": action, "argument": argument}


def build_prompt(case, role, transcript):
    history = "\n".join([
        f"{t['role'].upper()}: {t['content']}"
        for t in transcript
    ]) if transcript else "[Court is now in session. You speak first.]"

    evidence_key = f"{role}_evidence"
    your_evidence = "; ".join(case.get(evidence_key, []))

    return f"""=== CASE ===
{case['title']} | Charge: {case['charge']}
Facts: {case['facts']}
Your Evidence: {your_evidence}

=== TRANSCRIPT ===
{history}

=== YOUR TURN ===
You are the {role.upper()}. Make your argument."""


# ── MAIN EPISODE RUNNER ──────────────────────────────────────
def run_episode(case_selection, custom_case_text):
    """Run a full courtroom episode and yield updates progressively."""

    # Get case
    if case_selection == "🎲 Random Case":
        case = random.choice(CASES)
    elif case_selection == "✏️ Custom Case" and custom_case_text.strip():
        case = {
            "id": "CUSTOM",
            "title": "Custom Case",
            "category": "Custom",
            "charge": custom_case_text.strip(),
            "facts": custom_case_text.strip(),
            "prosecutor_evidence": ["Evidence submitted by prosecution"],
            "defense_evidence": ["Evidence submitted by defense"]
        }
    else:
        case = CASE_MAP.get(case_selection, random.choice(CASES))

    transcript = []
    p_rewards = []
    d_rewards = []

    # Header
    header = f"""⚖️ VERDICT — COURTROOM SESSION
{'═'*50}
📋 Case: {case['title']}
⚡ Charge: {case['charge']}
📜 Facts: {case['facts'][:200]}{'...' if len(case['facts']) > 200 else ''}
{'═'*50}
"""
    yield header, "", "", "🔄 Starting...", "—", "—", "—"
    time.sleep(0.5)

    full_transcript = header

    # Run rounds
    for round_num in range(1, MAX_ROUNDS + 1):
        round_header = f"\n🔔 ROUND {round_num}\n{'─'*40}\n"
        full_transcript += round_header
        yield full_transcript, "", "", f"⚖️ Round {round_num} in progress...", "—", "—", "—"

        for role in ["prosecutor", "defense"]:
            emoji = "⚔️" if role == "prosecutor" else "🛡️"
            label = role.upper()

            # Show thinking
            thinking = f"{emoji} {label} is arguing...\n"
            yield full_transcript + thinking, "", "", f"🤔 {label} thinking...", "—", "—", "—"

            # Generate
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(case, role, transcript)}
            ]
            raw = generate(messages, max_tokens=180, temperature=0.75)
            parsed = parse_response(raw)
            reward = score_argument(raw)

            # Log
            turn = {
                "role": role,
                "round": round_num,
                "action": parsed["action"],
                "content": parsed["argument"],
                "reward": reward
            }
            transcript.append(turn)

            if role == "prosecutor":
                p_rewards.append(reward)
            else:
                d_rewards.append(reward)

            # Format turn
            action_emoji = {
                "argue": "💬",
                "counter": "↩️",
                "present_evidence": "📎",
                "object": "🚫",
                "close": "🔒"
            }.get(parsed["action"], "💬")

            turn_text = f"""{emoji} {label} [{action_emoji} {parsed['action'].upper()}] (reward: {reward:+.3f})
{parsed['argument']}

"""
            full_transcript += turn_text
            yield full_transcript, "", "", f"✅ {label} argued | reward: {reward:+.3f}", "—", "—", "—"
            time.sleep(0.3)

    # Judge verdict
    yield full_transcript, "", "", "👨‍⚖️ Judge deliberating...", "—", "—", "—"

    judge_messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Case: {case['title']}\nCharge: {case['charge']}\n\nFull Transcript:\n" +
         "\n".join([f"{t['role'].upper()} ({t['action']}): {t['content']}" for t in transcript])}
    ]
    judge_raw = generate(judge_messages, max_tokens=200, temperature=0.3)

    # Parse verdict
    winner = "Draw"
    p_score = "—"
    d_score = "—"
    reasoning = judge_raw

    w = re.search(r'winner\s*:\s*(\w+)', judge_raw, re.IGNORECASE)
    if w: winner = w.group(1)

    ps = re.search(r'score_prosecutor\s*:\s*(\d+)', judge_raw, re.IGNORECASE)
    if ps: p_score = ps.group(1) + "/10"

    ds = re.search(r'score_defense\s*:\s*(\d+)', judge_raw, re.IGNORECASE)
    if ds: d_score = ds.group(1) + "/10"

    r = re.search(r'reasoning\s*:\s*(.+)', judge_raw, re.DOTALL | re.IGNORECASE)
    if r: reasoning = r.group(1).strip()

    # Final stats
    p_avg = sum(p_rewards) / len(p_rewards) if p_rewards else 0
    d_avg = sum(d_rewards) / len(d_rewards) if d_rewards else 0

    verdict_winner_emoji = {
        "Prosecutor": "⚔️ PROSECUTOR WINS",
        "Defense": "🛡️ DEFENSE WINS",
        "Draw": "🤝 DRAW"
    }.get(winner, "🤝 DRAW")

    verdict_text = f"""
{'═'*50}
👨‍⚖️ JUDGE'S VERDICT
{'═'*50}
🏆 {verdict_winner_emoji}

📊 Scores:
   Prosecutor: {p_score}
   Defense:    {d_score}

💭 Reasoning:
{reasoning}
{'═'*50}
📈 Avg Rewards:
   Prosecutor: {p_avg:+.3f}
   Defense:    {d_avg:+.3f}
"""
    full_transcript += verdict_text

    yield (
        full_transcript,
        verdict_winner_emoji,
        reasoning,
        "✅ Episode complete!",
        p_score,
        d_score,
        f"P: {p_avg:+.3f} | D: {d_avg:+.3f}"
    )


# ── GRADIO UI ────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --gold: #C9A84C;
    --dark: #0D0D0D;
    --darker: #080808;
    --panel: #141414;
    --border: #2A2A2A;
    --text: #E8E0D0;
    --muted: #888;
    --prosecutor: #E05252;
    --defense: #52A0E0;
    --judge: #C9A84C;
}

body, .gradio-container {
    background: var(--darker) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}

h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

.verdict-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.verdict-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: var(--gold);
    letter-spacing: 0.05em;
    margin: 0;
}

.verdict-subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.transcript-box textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
}

.stat-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.run-btn {
    background: var(--gold) !important;
    color: var(--dark) !important;
    font-weight: 700 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    border-radius: 4px !important;
    cursor: pointer !important;
}

.run-btn:hover {
    background: #D4B866 !important;
    transform: translateY(-1px);
}

label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

.gr-textbox, .gr-dropdown select {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}
"""

with gr.Blocks(css=css, title="Verdict ⚖️ Courtroom AI") as demo:

    gr.HTML("""
    <div class="verdict-header">
        <p class="verdict-title">⚖️ VERDICT</p>
        <p class="verdict-subtitle">Multi-Agent Courtroom AI · Trained with GRPO · OpenEnv Hackathon 2026</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Case Selection")

            case_options = ["🎲 Random Case", "✏️ Custom Case"] + CASE_TITLES
            case_dropdown = gr.Dropdown(
                choices=case_options,
                value="🎲 Random Case",
                label="Select a Case",
                interactive=True
            )

            custom_case = gr.Textbox(
                label="Custom Case Description (if selected above)",
                placeholder="Describe your own case...",
                lines=3,
                visible=False
            )

            run_btn = gr.Button("⚖️ Begin Trial", elem_classes=["run-btn"])

            gr.Markdown("### 📊 Live Stats")
            with gr.Row():
                p_score_box = gr.Textbox(label="Prosecutor Score", value="—", interactive=False)
                d_score_box = gr.Textbox(label="Defense Score", value="—", interactive=False)

            reward_box  = gr.Textbox(label="Avg Rewards", value="—", interactive=False)
            status_box  = gr.Textbox(label="Status", value="Ready", interactive=False)

            gr.Markdown("### 🏆 Verdict")
            winner_box    = gr.Textbox(label="Winner", value="—", interactive=False)
            reasoning_box = gr.Textbox(label="Judge's Reasoning", value="—", lines=4, interactive=False)

            gr.Markdown("""
---
**Model:** [Imaginephoenix/verdict-qwen2.5-3b-grpo](https://huggingface.co/Imaginephoenix/verdict-qwen2.5-3b-grpo)  
**Environment:** OpenEnv · GRPO via HuggingFace TRL  
**Hackathon:** Meta × HuggingFace × PyTorch 2026
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 📜 Courtroom Transcript")
            transcript_box = gr.Textbox(
                label="",
                value="Select a case and press 'Begin Trial' to start...",
                lines=35,
                interactive=False,
                elem_classes=["transcript-box"]
            )

    # Show/hide custom case
    def toggle_custom(choice):
        return gr.update(visible=(choice == "✏️ Custom Case"))

    case_dropdown.change(toggle_custom, inputs=case_dropdown, outputs=custom_case)

    # Run episode
    run_btn.click(
        fn=run_episode,
        inputs=[case_dropdown, custom_case],
        outputs=[
            transcript_box,
            winner_box,
            reasoning_box,
            status_box,
            p_score_box,
            d_score_box,
            reward_box
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
