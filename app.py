"""
Verdict — Premium Gradio App (HuggingFace Spaces)
"""
import sys, os, random, json
from textwrap import dedent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
import gradio as gr
from models import ActionType, AgentRole, TrialPhase, VerdictAction, VerdictObservation, RubricScore
from verdict_environment import VerdictEnvironment, load_cases

ALL_CASES = load_cases()
EASY_CASES = load_cases(difficulty="easy")
MEDIUM_CASES = load_cases(difficulty="medium")
HARD_CASES = load_cases(difficulty="hard")

# ── Premium CSS ──────────────────────────────────────────────────────
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global Reset ── */
* { box-sizing: border-box; }
body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #06080f !important;
    color: #e2e8f0 !important;
}
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

/* ── Animated Hero ── */
.verdict-hero {
    position: relative;
    background: linear-gradient(160deg, #0a0f1e 0%, #111827 40%, #1a1033 100%);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 28px;
    padding: 48px 40px 40px;
    margin-bottom: 28px;
    overflow: hidden;
    box-shadow: 0 0 80px rgba(139,92,246,0.08), 0 30px 60px rgba(0,0,0,0.4);
}
.verdict-hero::before {
    content: '';
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(168,85,247,0.06) 0%, transparent 50%);
    animation: heroGlow 8s ease-in-out infinite alternate;
}
@keyframes heroGlow {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(2%, -2%) scale(1.05); }
}
.verdict-hero .hero-inner { position: relative; z-index: 2; }
.verdict-hero h1 {
    font-size: 3rem; font-weight: 900; margin: 0 0 6px;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em; line-height: 1.1;
}
.verdict-hero .tagline {
    font-size: 1.05rem; color: rgba(203,213,225,0.75); max-width: 640px;
    line-height: 1.6; margin: 0;
}
.verdict-hero .badge-row { display: flex; gap: 8px; margin-top: 18px; flex-wrap: wrap; }
.verdict-hero .badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.25);
    color: #c4b5fd; letter-spacing: 0.02em; backdrop-filter: blur(8px);
}
.verdict-hero .badge.green { background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,0.25); color: #86efac; }
.verdict-hero .badge.amber { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.25); color: #fcd34d; }
.verdict-hero .stats-row {
    display: flex; gap: 32px; margin-top: 24px; padding-top: 20px;
    border-top: 1px solid rgba(148,163,184,0.08);
}
.verdict-hero .stat { text-align: center; }
.verdict-hero .stat-num {
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.verdict-hero .stat-label { font-size: 0.7rem; color: rgba(148,163,184,0.6); text-transform: uppercase; letter-spacing: 0.1em; }

/* ── Tabs ── */
.gr-tabs { border: none !important; }
button.tab-nav { 
    font-weight: 600 !important; font-size: 0.9rem !important;
    border-radius: 12px 12px 0 0 !important;
    transition: all 0.25s ease !important;
}

/* ── Cards ── */
.glass-card {
    background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(20,27,48,0.88)) !important;
    border: 1px solid rgba(100,116,139,0.12) !important;
    border-radius: 20px !important; padding: 24px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.2) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
.glass-card:hover {
    border-color: rgba(139,92,246,0.25) !important;
    box-shadow: 0 8px 40px rgba(139,92,246,0.08) !important;
}

/* ── Transcript ── */
.transcript-box {
    background: linear-gradient(180deg, rgba(6,8,15,0.95), rgba(15,23,42,0.9)) !important;
    border: 1px solid rgba(100,116,139,0.15) !important;
    border-radius: 16px !important; padding: 24px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
    color: #cbd5e1 !important; max-height: 600px; overflow-y: auto;
    line-height: 1.7 !important;
}
.transcript-box h2 { color: #a78bfa !important; }
.transcript-box strong { color: #e2e8f0 !important; }
.transcript-box code {
    background: rgba(139,92,246,0.15) !important; color: #c4b5fd !important;
    padding: 2px 8px !important; border-radius: 6px !important; font-size: 0.8rem !important;
}
.transcript-box blockquote {
    border-left: 3px solid rgba(139,92,246,0.4) !important;
    padding-left: 14px !important; margin: 8px 0 !important;
    color: rgba(203,213,225,0.8) !important; font-style: italic !important;
}

/* ── Rubric Scores ── */
.score-card {
    background: linear-gradient(145deg, rgba(15,20,35,0.95), rgba(20,25,50,0.9)) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 16px !important; padding: 20px !important; color: #e2e8f0 !important;
}
.score-card strong { color: #a78bfa !important; }

/* ── Buttons ── */
.run-trial-btn {
    background: linear-gradient(135deg, #7c3aed, #6d28d9, #5b21b6) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 14px !important; padding: 14px 32px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    color: white !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    transition: all 0.3s ease !important;
}
.run-trial-btn:hover {
    box-shadow: 0 6px 30px rgba(124,58,237,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Form Fields ── */
.gr-input, .gr-dropdown, textarea, select {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(100,116,139,0.2) !important;
    border-radius: 12px !important; color: #e2e8f0 !important;
    transition: border-color 0.2s ease !important;
}
.gr-input:focus, textarea:focus {
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
}
label { color: #94a3b8 !important; font-weight: 600 !important; font-size: 0.85rem !important; }

/* ── Evidence Colors ── */
.evidence-prosecutor { color: #f87171 !important; }
.evidence-defense { color: #60a5fa !important; }

/* ── About Page ── */
.about-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-top: 16px; }
.about-item {
    background: rgba(15,23,42,0.7); border: 1px solid rgba(100,116,139,0.1);
    border-radius: 16px; padding: 20px; text-align: center;
    transition: all 0.3s ease;
}
.about-item:hover { border-color: rgba(139,92,246,0.3); transform: translateY(-2px); }
.about-item .icon { font-size: 2rem; margin-bottom: 8px; }
.about-item h4 { margin: 0 0 4px; color: #e2e8f0; font-weight: 700; }
.about-item p { margin: 0; color: rgba(148,163,184,0.7); font-size: 0.82rem; line-height: 1.5; }

/* ── Rubric Table ── */
.rubric-section table { width: 100%; border-collapse: separate; border-spacing: 0; }
.rubric-section th {
    background: rgba(139,92,246,0.12) !important; color: #c4b5fd !important;
    padding: 12px 16px !important; font-weight: 700 !important; text-align: left;
    border-bottom: 2px solid rgba(139,92,246,0.2) !important;
}
.rubric-section td {
    padding: 10px 16px !important; border-bottom: 1px solid rgba(100,116,139,0.08) !important;
    color: #cbd5e1 !important;
}
"""

# ── Logic ─────────────────────────────────────────────────────────────

def format_case_list():
    choices = []
    for c in ALL_CASES:
        tier = c.get("difficulty", "medium").upper()
        icon = {"EASY": "🟢", "MEDIUM": "🟡", "HARD": "🔴"}.get(tier, "⚪")
        choices.append(f"{icon} [{tier}] {c['case_id']}: {c['charge']}")
    return choices

def get_case_details(selection: str) -> str:
    if not selection: return ""
    case_id = selection.split(":")[0].split("]")[-1].strip()
    case = next((c for c in ALL_CASES if c["case_id"] == case_id), None)
    if not case: return "Case not found."
    p_ev = "\n".join(f"  - 🔴 **`{e.evidence_id}`** {e.description}" for e in case["prosecutor_evidence"])
    d_ev = "\n".join(f"  - 🔵 **`{e.evidence_id}`** {e.description}" for e in case["defense_evidence"])
    diff_badge = {"easy": "🟢 EASY", "medium": "🟡 MEDIUM", "hard": "🔴 HARD"}.get(case.get("difficulty","medium"), "⚪")
    return dedent(f"""\
### 📋 {case.get('title', case['case_id'])}
**{case.get('category', 'N/A')}** · {diff_badge}

---

**📜 Case Facts:**
> {case['case_brief']}

---

**🔴 Prosecution Evidence:**
{p_ev}

**🔵 Defense Evidence:**
{d_ev}
""")

def run_simulation(selection, user_role, user_argument):
    if not selection or not user_argument.strip():
        return "### ⚠️ Please select a case and write your argument.", ""
    case_id = selection.split(":")[0].split("]")[-1].strip()
    case = next((c for c in ALL_CASES if c["case_id"] == case_id), None)
    if not case: return "Case not found.", ""

    env = VerdictEnvironment(max_rounds=2)
    env.reset()
    s = env.state
    s.case_id, s.case_brief, s.charge = case["case_id"], case["case_brief"], case["charge"]
    s.prosecutor_evidence = [e.model_copy() for e in case["prosecutor_evidence"]]
    s.defense_evidence = [e.model_copy() for e in case["defense_evidence"]]

    lines, scores = [], []
    user_is_pros = user_role == "🔴 Prosecutor"
    step = 0

    def bot_action(phase):
        at = {TrialPhase.CLOSING_STATEMENTS: ActionType.CLOSE}.get(phase, ActionType.ARGUE)
        return VerdictAction(thinking="Counter-argument strategy.", action_type=at,
            argument="The evidence does not support opposing claims. The documented facts reveal a different conclusion. The timeline contradicts the opposing narrative. The court should weigh these facts carefully.")

    def user_action(phase):
        at = {TrialPhase.CLOSING_STATEMENTS: ActionType.CLOSE}.get(phase, ActionType.ARGUE)
        return VerdictAction(thinking="Strategic reasoning.", action_type=at, argument=user_argument)

    while not env.state.is_done and step < 20:
        role = env.state.current_speaker
        is_user = (role == AgentRole.PROSECUTOR) == user_is_pros
        action = user_action(env.state.phase) if is_user else bot_action(env.state.phase)
        label = f"{'🧑 YOU' if is_user else '🤖 AI'} ({role.value.upper()})"
        result = env.step(action)
        step += 1
        phase_name = env.state.phase.value.replace("_", " ").title()
        lines.append(f"### Step {step} · {phase_name}\n**{label}** · Action: `{action.action_type.value}`\n\n> {action.argument[:300]}\n\n**Reward:** `{result.reward:.3f}`\n")
        if result.reward_breakdown:
            rb = result.reward_breakdown
            scores.append(f"| {step} | {role.value.title()} | {rb.coherence:.2f} | {rb.evidence_usage:.2f} | {rb.counter_quality:.2f} | {rb.consistency:.2f} | **{rb.weighted_total:.3f}** |")

    s = env.state
    verdict_text = s.verdict or "No verdict reached."
    winner = s.winner.value.upper() if s.winner else "DRAW"
    p_avg = (sum(r.weighted_total for r in s.prosecutor_scores) / len(s.prosecutor_scores)) if s.prosecutor_scores else 0
    d_avg = (sum(r.weighted_total for r in s.defense_scores) / len(s.defense_scores)) if s.defense_scores else 0

    transcript = "\n---\n\n".join(lines)
    transcript += f"\n\n---\n\n## ⚖️ VERDICT\n\n**{verdict_text}**\n\n"
    transcript += f"| | Score |\n|---|---|\n| 🔴 Prosecution Avg | **{p_avg:.3f}** |\n| 🔵 Defense Avg | **{d_avg:.3f}** |\n| Winner | **{winner}** |\n| Total Steps | {s.step_count} |"

    score_table = "| Step | Role | Coherence | Evidence | Counter | Consistency | Total |\n|------|------|-----------|----------|---------|-------------|-------|\n"
    score_table += "\n".join(scores) if scores else "| — | — | — | — | — | — | — |"
    return transcript, score_table

# ── UI ────────────────────────────────────────────────────────────────

def build_interface():
    with gr.Blocks(css=APP_CSS, title="Verdict ⚖️ Courtroom AI", theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    )) as demo:

        gr.HTML(f"""
        <div class='verdict-hero'>
            <div class='hero-inner'>
                <h1>⚖️ Verdict</h1>
                <p class='tagline'>
                    Multi-agent adversarial courtroom simulation — argue real legal cases against an AI opponent,
                    evaluated by a 5-dimension composable rubric. Built on OpenEnv for the
                    <strong>Meta × HuggingFace × PyTorch Hackathon</strong>.
                </p>
                <div class='badge-row'>
                    <span class='badge'>🧠 OpenEnv v0.2.3</span>
                    <span class='badge green'>🔥 PyTorch + TRL</span>
                    <span class='badge amber'>🏆 Hackathon Submission</span>
                    <span class='badge'>⚡ GRPO Training</span>
                </div>
                <div class='stats-row'>
                    <div class='stat'><div class='stat-num'>{len(ALL_CASES)}</div><div class='stat-label'>Cases</div></div>
                    <div class='stat'><div class='stat-num'>5</div><div class='stat-label'>Rubric Dims</div></div>
                    <div class='stat'><div class='stat-num'>3</div><div class='stat-label'>Difficulty Tiers</div></div>
                    <div class='stat'><div class='stat-num'>6</div><div class='stat-label'>Action Types</div></div>
                </div>
            </div>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("⚔️ Play Trial"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=340):
                        gr.HTML("<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:8px'>📋 Case Selection</div>")
                        case_dd = gr.Dropdown(label="Select Case", choices=format_case_list(),
                            value=format_case_list()[0] if ALL_CASES else None)
                        case_info = gr.Markdown(get_case_details(format_case_list()[0]) if ALL_CASES else "",
                            elem_classes=["glass-card"])
                    with gr.Column(scale=2):
                        gr.HTML("<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin-bottom:8px'>⚔️ Your Move</div>")
                        role = gr.Radio(label="Choose Your Side", choices=["🔴 Prosecutor", "🔵 Defense"], value="🔴 Prosecutor")
                        arg = gr.Textbox(label="Courtroom Argument", placeholder="Present your case to the court. Reference evidence by ID for scoring bonus...", lines=5)
                        btn = gr.Button("⚖️ Run Trial", variant="primary", size="lg", elem_classes=["run-trial-btn"])
                        gr.HTML("<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin:20px 0 8px'>📜 Trial Transcript</div>")
                        transcript = gr.Markdown("*Select a case, pick your side, and present your argument.*", elem_classes=["transcript-box"])
                        gr.HTML("<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-weight:700;margin:16px 0 8px'>📊 Rubric Breakdown</div>")
                        rubric_out = gr.Markdown("", elem_classes=["score-card"])
                case_dd.change(get_case_details, case_dd, case_info)
                btn.click(run_simulation, [case_dd, role, arg], [transcript, rubric_out])

            with gr.Tab("📋 Cases"):
                gr.HTML(f"""
                <div style='display:flex;gap:16px;margin-bottom:20px;flex-wrap:wrap'>
                    <div class='about-item' style='flex:1;min-width:120px'><div class='icon'>🟢</div><h4>{len(EASY_CASES)}</h4><p>Easy Cases</p></div>
                    <div class='about-item' style='flex:1;min-width:120px'><div class='icon'>🟡</div><h4>{len(MEDIUM_CASES)}</h4><p>Medium Cases</p></div>
                    <div class='about-item' style='flex:1;min-width:120px'><div class='icon'>🔴</div><h4>{len(HARD_CASES)}</h4><p>Hard Cases</p></div>
                </div>""")
                bdd = gr.Dropdown(label="Browse All Cases", choices=format_case_list(), value=format_case_list()[0] if ALL_CASES else None)
                binfo = gr.Markdown(get_case_details(format_case_list()[0]) if ALL_CASES else "", elem_classes=["glass-card"])
                bdd.change(get_case_details, bdd, binfo)

            with gr.Tab("📊 Rubric"):
                gr.HTML("""<div class='rubric-section'>
                <h2 style='color:#c4b5fd;margin-bottom:4px'>⚖️ The 5-Dimension Composable Rubric</h2>
                <p style='color:#64748b;margin-bottom:20px'>Dense reward signal — prevents reward hacking via structured multi-axis evaluation.</p>
                <table>
                <tr><th>Component</th><th>Weight</th><th>Measures</th><th>Anti-Gaming</th></tr>
                <tr><td>🧠 <strong>Coherence</strong></td><td>30%</td><td>Logical flow, structured sentences, case-fact grounding</td><td>Verbose but hollow → low score</td></tr>
                <tr><td>📎 <strong>Evidence Usage</strong></td><td>20%</td><td>Strategic citation of evidence cards</td><td>Evidence dumping → near-zero</td></tr>
                <tr><td>⚔️ <strong>Counter Quality</strong></td><td>20%</td><td>Directly addresses opponent's strongest point</td><td>Talking past opponent → penalty</td></tr>
                <tr><td>🔄 <strong>Consistency</strong></td><td>15%</td><td>No self-contradiction across turns</td><td>Repetitive loops → heavy penalty</td></tr>
                <tr><td>🏛️ <strong>Verdict Alignment</strong></td><td>15%</td><td>Terminal bonus — did the judge rule in your favor?</td><td>Binary 0/1 at episode end</td></tr>
                </table>
                <div style='margin-top:20px;padding:16px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.15);border-radius:12px'>
                    <strong style='color:#a78bfa'>Format Penalties:</strong>
                    <span style='color:#94a3b8'> Missing XML tags → <code style="color:#f87171">-1.0</code> · Over 300 words → <code style="color:#f87171">-1.0</code> · Invalid action → <code style="color:#fbbf24">-0.5</code></span>
                </div></div>""")

            with gr.Tab("ℹ️ About"):
                gr.HTML("""
                <h2 style='color:#e2e8f0;margin-bottom:4px'>Verdict — Multi-Agent Courtroom RL</h2>
                <p style='color:#64748b;margin-bottom:24px'>Training LLMs to reason, argue, and judge like a lawyer.</p>
                <div class='about-grid'>
                    <div class='about-item'><div class='icon'>🏛️</div><h4>Courtroom POMDP</h4><p>Partially observable — each side has hidden evidence cards the opponent can't see.</p></div>
                    <div class='about-item'><div class='icon'>⚔️</div><h4>Adversarial RL</h4><p>Prosecutor vs Defense in zero-sum argumentation with strategic evidence reveals.</p></div>
                    <div class='about-item'><div class='icon'>🧠</div><h4>Theory of Mind</h4><p>Agents must reason about opponent's hidden evidence and anticipate counter-arguments.</p></div>
                    <div class='about-item'><div class='icon'>📊</div><h4>Dense Rewards</h4><p>5-dimension composable rubric prevents reward hacking and sparse signal problems.</p></div>
                    <div class='about-item'><div class='icon'>🔥</div><h4>GRPO Training</h4><p>Group Relative Policy Optimization via HuggingFace TRL + Unsloth for fast fine-tuning.</p></div>
                    <div class='about-item'><div class='icon'>🌐</div><h4>OpenEnv v0.2.3</h4><p>Built on the official OpenEnv framework for standardized multi-agent environments.</p></div>
                </div>
                <div style='margin-top:24px;padding:20px;background:rgba(15,23,42,0.7);border:1px solid rgba(100,116,139,0.1);border-radius:16px'>
                    <strong style='color:#a78bfa'>Meta × HuggingFace × PyTorch Hackathon — Bangalore 2026</strong>
                    <p style='color:#64748b;margin:8px 0 0'>Top 200 of 32,000 teams selected. Building AI that challenges, not agrees.</p>
                </div>""")
    return demo

if __name__ == "__main__":
    build_interface().launch(server_name="0.0.0.0", server_port=7860)
