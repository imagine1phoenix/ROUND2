"""
Verdict — Gradio Demo (HF Spaces)
===================================
Wired into the real VerdictEnvironment. Runs live courtroom episodes
with composable rubric scoring. Built for Hugging Face Spaces deployment.
"""

import sys
import os
import random
import json
from textwrap import dedent

# Wire imports to our server modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr

from models import (
    ActionType, AgentRole, TrialPhase,
    VerdictAction, VerdictObservation, RubricScore,
)
from verdict_environment import VerdictEnvironment, load_cases

# ---------------------------------------------------------------------------
#  Load real cases from cases.json
# ---------------------------------------------------------------------------

ALL_CASES = load_cases()
EASY_CASES = load_cases(difficulty="easy")
MEDIUM_CASES = load_cases(difficulty="medium")
HARD_CASES = load_cases(difficulty="hard")

# Global env instance for the demo
ENV = VerdictEnvironment(max_rounds=2)

RUBRIC_TEXT = dedent("""\
### ⚖️ Judge Rubric (Composable, Dense Signal)

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Argument Coherence** | 30% | Logical flow, structured sentences, case-fact grounding |
| **Evidence Usage** | 20% | Strategic citation of evidence; penalizes evidence dumping |
| **Counter Quality** | 20% | Directly addresses opponent's strongest point |
| **Consistency** | 15% | No self-contradiction or repetition across turns |
| **Verdict Alignment** | 15% | Terminal bonus — did the judge rule in your favor? |

> Anti-gaming: verbose but hollow arguments → low coherence.
> Repeating the same point → consistency penalty.
> Evidence dump without narrative → near-zero evidence score.
""")

# ---------------------------------------------------------------------------
#  CSS (preserved from meta branch + enhanced)
# ---------------------------------------------------------------------------

APP_CSS = """
body { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

.hero-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.92));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 24px 80px rgba(15,23,42,0.18);
    color: #f8fafc;
}

.hero-card h1 {
    font-size: 2.4rem;
    margin: 0 0 8px 0;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.transcript-box {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 16px;
    padding: 20px;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.88rem;
    max-height: 500px;
    overflow-y: auto;
}

.score-card {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 16px;
    padding: 18px;
    color: #e2e8f0;
}

.gr-button {
    border-radius: 999px !important;
    font-weight: 700 !important;
}
"""

# ---------------------------------------------------------------------------
#  Core Demo Logic — runs real episodes
# ---------------------------------------------------------------------------

def format_case_list():
    """Build dropdown choices from real cases."""
    choices = []
    for c in ALL_CASES:
        tier = c.get("difficulty", "medium").upper()
        choices.append(f"[{tier}] {c['case_id']}: {c['charge']}")
    return choices


def get_case_details(selection: str) -> str:
    """Show full case brief + evidence for the selected case."""
    if not selection:
        return "Select a case to view details."
    case_id = selection.split(":")[0].split("]")[-1].strip()
    case = next((c for c in ALL_CASES if c["case_id"] == case_id), None)
    if not case:
        return "Case not found."

    p_ev = "\n".join(f"  - **[{e.evidence_id}]** {e.title}: {e.description}" for e in case["prosecutor_evidence"])
    d_ev = "\n".join(f"  - **[{e.evidence_id}]** {e.title}: {e.description}" for e in case["defense_evidence"])

    return dedent(f"""\
### {case.get('title', case['case_id'])}
**Category:** {case.get('category', 'N/A')} · **Difficulty:** {case.get('difficulty', 'medium').upper()}

**Facts:**
{case['case_brief']}

**🔴 Prosecutor Evidence (hidden from defense):**
{p_ev}

**🔵 Defense Evidence (hidden from prosecutor):**
{d_ev}
""")


def run_simulation(selection: str, user_role: str, user_argument: str) -> tuple[str, str]:
    """Run a full episode. The user plays one side, rule-based agent plays the other."""
    if not selection or not user_argument.strip():
        return "⚠️ Select a case and enter your argument.", ""

    case_id = selection.split(":")[0].split("]")[-1].strip()
    case = next((c for c in ALL_CASES if c["case_id"] == case_id), None)
    if not case:
        return "Case not found.", ""

    # Create a fresh env with this specific case
    env = VerdictEnvironment(max_rounds=2)
    obs = env.reset()
    # Override with selected case
    s = env.state
    s.case_id = case["case_id"]
    s.case_brief = case["case_brief"]
    s.charge = case["charge"]
    s.prosecutor_evidence = [e.model_copy() for e in case["prosecutor_evidence"]]
    s.defense_evidence = [e.model_copy() for e in case["defense_evidence"]]

    transcript_lines = []
    scores_lines = []
    user_is_prosecutor = user_role == "Prosecutor"
    step = 0

    def bot_action(phase: TrialPhase) -> VerdictAction:
        """Simple rule-based opponent."""
        phase_map = {
            TrialPhase.PLEA_BARGAIN: ActionType.ARGUE,
            TrialPhase.OPENING_STATEMENTS: ActionType.ARGUE,
            TrialPhase.ARGUMENT_ROUNDS: ActionType.ARGUE,
            TrialPhase.CLOSING_STATEMENTS: ActionType.CLOSE,
        }
        at = phase_map.get(phase, ActionType.ARGUE)
        return VerdictAction(
            thinking="I must present a strong counter-argument addressing the key facts.",
            action_type=at,
            argument=(
                "The evidence presented does not support the claims made. "
                "However, upon examining the documented facts, a different conclusion emerges. "
                "Furthermore, the timeline of events contradicts the opposing narrative. "
                "Therefore, the court should consider these facts carefully before ruling."
            ),
        )

    def user_action(phase: TrialPhase) -> VerdictAction:
        phase_map = {
            TrialPhase.PLEA_BARGAIN: ActionType.ARGUE,
            TrialPhase.OPENING_STATEMENTS: ActionType.ARGUE,
            TrialPhase.ARGUMENT_ROUNDS: ActionType.ARGUE,
            TrialPhase.CLOSING_STATEMENTS: ActionType.CLOSE,
        }
        at = phase_map.get(phase, ActionType.ARGUE)
        return VerdictAction(
            thinking="Strategic reasoning based on the available evidence.",
            action_type=at,
            argument=user_argument,
        )

    # Run the episode
    while not env.state.is_done and step < 20:
        role = env.state.current_speaker
        is_user_turn = (role == AgentRole.PROSECUTOR) == user_is_prosecutor

        if is_user_turn:
            action = user_action(env.state.phase)
            label = f"🧑 YOU ({role.value.upper()})"
        else:
            action = bot_action(env.state.phase)
            label = f"🤖 BOT ({role.value.upper()})"

        result = env.step(action)
        step += 1

        transcript_lines.append(
            f"**Step {step} · {env.state.phase.value} · {label}**\n"
            f"Action: `{action.action_type.value}`\n"
            f"> {action.argument[:300]}\n"
            f"Reward: `{result.reward:.3f}`\n"
        )

        if result.observation.reward_breakdown:
            rb = result.observation.reward_breakdown
            scores_lines.append(
                f"Step {step} ({role.value}): "
                f"coh={rb.coherence:.2f} ev={rb.evidence_usage:.2f} "
                f"ctr={rb.counter_quality:.2f} con={rb.consistency:.2f} "
                f"→ **{rb.weighted_total:.3f}**"
            )

    # Final verdict
    s = env.state
    verdict_text = s.verdict or "No verdict reached."
    winner = s.winner.value.upper() if s.winner else "DRAW"

    p_avg = (sum(r.weighted_total for r in s.prosecutor_scores) / len(s.prosecutor_scores)) if s.prosecutor_scores else 0
    d_avg = (sum(r.weighted_total for r in s.defense_scores) / len(s.defense_scores)) if s.defense_scores else 0

    transcript = "\n---\n".join(transcript_lines)
    transcript += f"\n\n---\n## ⚖️ VERDICT\n**{verdict_text}**\n\n**Winner:** {winner}\n"
    transcript += f"**Prosecution avg:** {p_avg:.3f} · **Defense avg:** {d_avg:.3f}\n"
    transcript += f"**Steps:** {s.step_count} · **Transcript entries:** {len(s.transcript)}"

    rubric_detail = "\n".join(scores_lines) if scores_lines else "No scores recorded."

    return transcript, rubric_detail


# ---------------------------------------------------------------------------
#  Gradio UI
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    with gr.Blocks(css=APP_CSS, title="Verdict ⚖️ Courtroom AI") as demo:
        # Hero
        gr.HTML("""
        <div class='hero-card'>
            <h1>⚖️ Verdict</h1>
            <p style='font-size:1.1rem; opacity:0.85; max-width:720px; margin:0;'>
                Multi-Agent Courtroom RL Environment — argue a case against an AI opponent,
                scored by a composable rubric. Built on OpenEnv for the Meta × HuggingFace Hackathon.
            </p>
        </div>
        """)

        with gr.Tabs():
            # --- Tab 1: Play a Case ---
            with gr.Tab("🎮 Play a Case"):
                with gr.Row():
                    with gr.Column(scale=1):
                        case_dropdown = gr.Dropdown(
                            label="Select Case",
                            choices=format_case_list(),
                            value=format_case_list()[0] if ALL_CASES else None,
                        )
                        case_details = gr.Markdown(
                            get_case_details(format_case_list()[0]) if ALL_CASES else "",
                            elem_classes=["score-card"],
                        )

                    with gr.Column(scale=2):
                        role_choice = gr.Radio(
                            label="Your Role",
                            choices=["Prosecutor", "Defense"],
                            value="Prosecutor",
                        )
                        user_input = gr.Textbox(
                            label="Your Argument",
                            placeholder="Write your courtroom argument here. Reference evidence by title for bonus points...",
                            lines=5,
                        )
                        run_btn = gr.Button("⚖️ Run Trial", variant="primary", size="lg")

                        transcript_output = gr.Markdown(
                            "_Select a case, choose your role, write your argument, and click 'Run Trial'._",
                            elem_classes=["transcript-box"],
                        )
                        rubric_output = gr.Markdown("", elem_classes=["score-card"])

                case_dropdown.change(get_case_details, case_dropdown, case_details)
                run_btn.click(
                    run_simulation,
                    inputs=[case_dropdown, role_choice, user_input],
                    outputs=[transcript_output, rubric_output],
                )

            # --- Tab 2: Case Browser ---
            with gr.Tab("📋 Case Browser"):
                gr.Markdown(f"## 30 Cases · 3 Difficulty Tiers\n\n"
                            f"**Easy:** {len(EASY_CASES)} · **Medium:** {len(MEDIUM_CASES)} · **Hard:** {len(HARD_CASES)}")
                browser_dropdown = gr.Dropdown(
                    label="Browse Cases",
                    choices=format_case_list(),
                    value=format_case_list()[0] if ALL_CASES else None,
                )
                browser_details = gr.Markdown(
                    get_case_details(format_case_list()[0]) if ALL_CASES else "",
                )
                browser_dropdown.change(get_case_details, browser_dropdown, browser_details)

            # --- Tab 3: Rubric ---
            with gr.Tab("📊 Judge Rubric"):
                gr.Markdown(RUBRIC_TEXT)

            # --- Tab 4: About ---
            with gr.Tab("ℹ️ About"):
                gr.Markdown(dedent("""\
                ## Verdict — Multi-Agent Courtroom RL Environment

                **Hackathon:** Meta × HuggingFace × PyTorch OpenEnv (India 2026)

                **What it does:**
                - Prosecutor and Defense agents argue real legal cases
                - Each action is scored by a 5-dimension composable rubric
                - Partial observability: each side has hidden evidence cards
                - Trained via GRPO self-play to beat sycophancy

                **Tech Stack:**
                - [OpenEnv](https://github.com/meta-pytorch/OpenEnv) v0.2.3
                - HuggingFace TRL (GRPO training)
                - Unsloth (fast fine-tuning)
                - Gradio (this demo)

                **Case Bank:** 30 cases across 30 legal categories
                - Easy (8) · Medium (11) · Hard (11)

                **Environment API:**
                ```python
                env = VerdictEnvironment(difficulty='hard')
                obs = env.reset()
                result = env.step(action)
                # result.observation, result.reward, result.done
                ```
                """))

    return demo


if __name__ == "__main__":
    build_interface().launch(server_name="0.0.0.0", server_port=7860)
