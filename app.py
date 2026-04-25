import gradio as gr
import random
from textwrap import dedent

CASES = [
    {
        "title": "Contract Dispute: Missing Deliverable",
        "summary": "A startup alleges that a consulting firm failed to deliver a promised delivery timeline, while the firm claims the client changed scope mid-project.",
        "evidence": [
            "Email thread confirming the scope was fixed on March 8.",
            "Internal note indicating an extra feature request after the first draft.",
            "Invoice showing payment for milestone 1 but not milestone 2."
        ],
        "theme": "Breach of contract, scope creep, and evidence timing."
    },
    {
        "title": "Wrongful Termination: Performance vs. Policy",
        "summary": "An employee alleges wrongful termination after raising ethical concerns, while the employer cites repeated policy violations.",
        "evidence": [
            "Performance reviews labeling the employee as ""a strong team player.""",
            "A confidential whistleblower report filed anonymously two months before termination.",
            "A written policy memo distributed company-wide on acceptable conduct."
        ],
        "theme": "Competing narratives between employee credibility and employer compliance."
    },
    {
        "title": "Ethical Breach: Confidentiality Leak",
        "summary": "A product manager is accused of sharing proprietary design details with a competitor, while they claim the information was already public.",
        "evidence": [
            "A timestamped message with the leaked file attached.",
            "A press release announcing a related product line one week earlier.",
            "A non-disclosure agreement signed at project start."
        ],
        "theme": "Balance between public domain facts and protected company secrets."
    }
]

SIMULATION_TEMPLATES = {
    "Prosecutor": [
        "The strongest path is to frame the case around the defendant's duty and the clear evidence timeline.",
        "Emphasize the burden of proof and how the opposing story contradicts the documented facts.",
        "Highlight the simplest narrative: evidence shows the contract terms were not honored."
    ],
    "Defense": [
        "Shift attention to the opponent's weak assumptions and any gaps in the evidence chain.",
        "Stress reasonable doubt and the fact that the disputed evidence can support multiple interpretations.",
        "Argue that the plaintiff's story ignores key mitigating context from the agreement."
    ],
    "Judge": [
        "A reasoned verdict must weigh the credibility of opposing claims and the strength of evidence.",
        "Focus on whether the legal standard of proof is met by the available facts.",
        "The final judgment should reward fairness over rhetorical flourish."
    ]
}

RUBRIC_TEXT = dedent(
    """
    ### Judge Rubric

    - **Argument Coherence (30%)** — logical flow, no self-contradictions, and clear connections between claim and support.
    - **Evidence Usage (20%)** — proper citation of facts; avoid evidence dumping.
    - **Counter Quality (20%)** — directly address the opponent's strongest point rather than deflecting.
    - **Consistency (15%)** — maintain a stable position across the trial.
    - **Verdict Alignment (15%)** — final reasoning must align with the legal standard and evidence record.

    > This demo is designed to highlight how a courtroom AI must think like a lawyer: structured, adversarial, and persuasive.
    """
)

LIGHT_DARK_CSS = """
body.theme-light {
    background: #eef2ff !important;
    color: #0f172a !important;
}
body.theme-dark {
    background: #020617 !important;
    color: #e2e8f0 !important;
}

.gradio-container {
    background: transparent !important;
}

.hero-card,
.card,
.gr-button,
.gradio-container .panel {
    border-radius: 24px !important;
}

.hero-card {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.18);
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 24px 80px rgba(15, 23, 42, 0.18);
}

body.theme-light .hero-card {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 18px 68px rgba(15, 23, 42, 0.08);
}

.card {
    background: rgba(8, 16, 36, 0.96);
    border: 1px solid rgba(148, 163, 184, 0.14);
    color: #f8fafc;
    padding: 22px;
}

body.theme-light .card {
    background: rgba(255, 255, 255, 0.96);
    border-color: rgba(148, 163, 184, 0.16);
    color: #111827;
}

.menu-card {
    background: rgba(15, 23, 42, 0.92);
    padding: 20px;
    border: 1px solid rgba(148, 163, 184, 0.16);
}

body.theme-light .menu-card {
    background: rgba(255, 255, 255, 0.96);
}

.theme-button {
    margin-right: 8px;
    border-radius: 999px;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
    font-weight: 700;
}

.theme-button.light {
    background: #f8fafc;
    color: #0f172a;
}

.theme-button.dark {
    background: #0f172a;
    color: #f8fafc;
}

.menu-label {
    font-size: 0.95rem;
    margin-bottom: 10px;
    display: block;
    font-weight: 700;
}

.gradio-textbox textarea,
.gradio-textarea textarea {
    background: rgba(15, 23, 42, 0.9) !important;
    color: #e2e8f0 !important;
}

body.theme-light .gradio-textbox textarea,
body.theme-light .gradio-textarea textarea {
    background: rgba(248, 250, 252, 0.98) !important;
    color: #0f172a !important;
}

.gr-button {
    border-radius: 999px !important;
    font-weight: 700 !important;
}
"""

GRADIO_JS = """
<script>
function setAppTheme(theme) {
    document.body.classList.remove('theme-light', 'theme-dark');
    document.body.classList.add('theme-' + theme);
}
window.addEventListener('DOMContentLoaded', function() {
    setAppTheme('dark');
});
</script>
"""

MENU_CONTENT = {
    "Overview": dedent(
        """
        ## Verdict: Courtroom AI Demo

        This frontend is built for Hugging Face Spaces and showcases a polished, modern interface for the `Verdict` multi-agent legal reasoning environment.

        Use the menu to explore case scenarios, simulate prosecutor or defense arguments, and compare judgment criteria in a clean, polished UI.
        """
    ),
    "Case Browser": "",
    "Argument Lab": "",
    "Judge Rubric": RUBRIC_TEXT,
}


def format_case(index: int) -> str:
    case = CASES[index]
    details = [f"- {item}" for item in case["evidence"]]
    return dedent(
        f"""
        ### {case['title']}

        **Summary:** {case['summary']}

        **Theme:** {case['theme']}

        **Evidence Highlights:**
        {chr(10).join(details)}
        """
    )


def render_panel(section: str) -> str:
    if section == "Case Browser":
        case_list = "\n".join([f"- **{i+1}. {case['title']}**" for i, case in enumerate(CASES)])
        return dedent(
            f"""
            ## Case Browser

            Select a case from the menu to review the current evidence and scenario.

            **Available scenarios:**
            {case_list}
            """
        )
    elif section == "Argument Lab":
        return dedent(
            """
            ## Argument Lab

            Compose a prompt and choose a role to see a polished courtroom-style response. The simulator helps visualize how each side builds persuasive reasoning.

            - Use the `Prosecutor` role to focus on evidence and legal burden.
            - Use the `Defense` role to stress doubt, alternative explanations, and policy context.
            - Use the `Judge` role to produce a reasoned verdict summary.
            """
        )
    return MENU_CONTENT.get(section, MENU_CONTENT["Overview"])


def update_menu(selection: str) -> str:
    return render_panel(selection)


def update_case_details(case_value: str) -> str:
    try:
        case_id = int(case_value.split(":", 1)[0]) - 1
    except Exception:
        return "Select a valid case to preview its facts and evidence."
    if 0 <= case_id < len(CASES):
        return format_case(case_id)
    return "Select a valid case to preview its facts and evidence."


def style_response(response: str) -> str:
    return f"### Simulated Response\n\n{response}"


def simulate_argument(prompt: str, role: str, case_value: str) -> str:
    if not prompt.strip():
        return "Please enter a clear issue or question to begin the simulation."
    try:
        case_id = int(case_value.split(":", 1)[0]) - 1
    except Exception:
        case_id = 0
    case = CASES[case_id if 0 <= case_id < len(CASES) else 0]
    template = random.choice(SIMULATION_TEMPLATES.get(role, SIMULATION_TEMPLATES["Prosecutor"]))
    return dedent(
        f"""
        **Role:** {role}

        **Scenario:** {case['title']}

        {template}

        **Prompt:** {prompt.strip()}

        **Sample reasoning:**
        - Reference the key evidence clearly.
        - Build each argument point in a structured sequence.
        - Maintain credibility by avoiding unsupported claims.
        """
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(css=LIGHT_DARK_CSS, title="Verdict: Courtroom Reasoning Demo") as demo:
        gr.HTML(f"""
        <div class='hero-card'>
            <h1>Verdict</h1>
            <p style='font-size:1.05rem; opacity:0.85; max-width:720px;'>
                A polished Hugging Face-ready courtroom interface for exploring adversarial legal reasoning, case scenarios, and judgment criteria.
            </p>
            <div style='margin-top:22px;'>
                <button class='theme-button light' onclick="setAppTheme('light')">Light Mode</button>
                <button class='theme-button dark' onclick="setAppTheme('dark')">Dark Mode</button>
            </div>
        </div>
        {GRADIO_JS}
        """)

        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Menu")
                menu = gr.Radio(
                    label="Navigation",
                    choices=["Overview", "Case Browser", "Argument Lab", "Judge Rubric"],
                    value="Overview",
                    interactive=True,
                )
                case_selector = gr.Dropdown(
                    label="Choose a case",
                    choices=[f"{i+1}: {case['title']}" for i, case in enumerate(CASES)],
                    value="1: " + CASES[0]["title"],
                )
                gr.Markdown("---")
                gr.Markdown("### Quick Tips")
                gr.Markdown(
                    "- Put all action items inside the menu.\n- Use the case browser for evidence-driven storytelling.\n- Leverage the argument lab to compare roles."
                )
            with gr.Column(scale=2, min_width=360):
                content = gr.Markdown(render_panel("Overview"), elem_id="main-content")
                case_details = gr.Markdown(format_case(0), elem_id="case-details")
                with gr.Accordion("Argument Simulator", open=True):
                    prompt_input = gr.Textbox(
                        label="Describe the issue or point to argue",
                        placeholder="For example: 'Explain why the evidence proves breach of contract.'",
                        lines=3,
                    )
                    role_choice = gr.Radio(
                        label="Choose role",
                        choices=["Prosecutor", "Defense", "Judge"],
                        value="Prosecutor",
                        interactive=True,
                    )
                    simulate_btn = gr.Button("Simulate Response")
                    simulation_output = gr.Markdown("_Submit a prompt to see a polished simulated argument._")

        menu.change(update_menu, menu, content)
        case_selector.change(update_case_details, case_selector, case_details)
        simulate_btn.click(simulate_argument, inputs=[prompt_input, role_choice, case_selector], outputs=simulation_output)

    return demo

if __name__ == "__main__":
    build_interface().launch(server_name="0.0.0.0", server_port=7860)
