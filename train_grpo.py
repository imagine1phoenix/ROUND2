"""
Verdict — GRPO Self-Play Training Script
==========================================
Trains courtroom LLM agents via Group Relative Policy Optimization.
Uses Unsloth for 4-bit LoRA fine-tuning + TRL GRPOTrainer.

Run in Google Colab (T4 GPU) or locally with CUDA.

Usage:
    python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
    python train_grpo.py --difficulty easy --rounds 2  # quick test
"""

import sys
import os
import json
import re
import random
import argparse
from typing import Optional

# Ensure server/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from models import (
    ActionType, AgentRole, TrialPhase,
    EvidenceCard, TranscriptEntry, RubricScore,
    VerdictAction, VerdictObservation, VerdictState,
)
from verdict_environment import VerdictEnvironment, load_cases


# ═══════════════════════════════════════════════════════════════════════════
#  1. SYSTEM PROMPT (Courtroom Agent Instructions)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a courtroom agent in a legal trial simulation.
You receive: case brief, your role (Prosecutor/Defense), trial phase,
the transcript so far, your private evidence, and public evidence.

Respond with VALID XML:
<thinking>
[Internal strategy: opponent's weakness, hidden evidence theory, reveal timing]
</thinking>
<action>[PLEA | ARGUE | OBJECT | REVEAL_EVIDENCE | CONCEDE | CLOSE]</action>
<argument>
[Your statement to the court. Max 200 words. Reference evidence by title.]
</argument>

RULES:
- plea_bargain phase: PLEA (accept deal) or ARGUE (reject, go to trial)
- opening_statements: ARGUE only
- argument_rounds: ARGUE, OBJECT, REVEAL_EVIDENCE, or CONCEDE
- closing_statements: CLOSE only
- REVEAL_EVIDENCE requires evidence_id tag: <evidence_id>P1</evidence_id>
- thinking is private. argument is public.
- Reference evidence titles in argument for evidence_usage score.
- Address opponent's last argument for counter_quality score.
- Do NOT repeat yourself (consistency penalty)."""


# ═══════════════════════════════════════════════════════════════════════════
#  2. PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

PHASE_ACTIONS = {
    TrialPhase.PLEA_BARGAIN: ["PLEA", "ARGUE"],
    TrialPhase.OPENING_STATEMENTS: ["ARGUE"],
    TrialPhase.ARGUMENT_ROUNDS: ["ARGUE", "OBJECT", "REVEAL_EVIDENCE", "CONCEDE"],
    TrialPhase.CLOSING_STATEMENTS: ["CLOSE"],
}


def build_observation_prompt(
    case: dict,
    role: AgentRole,
    phase: TrialPhase,
    turn: int = 0,
    transcript: list[TranscriptEntry] | None = None,
    public_evidence: list[EvidenceCard] | None = None,
) -> str:
    """Build a courtroom observation prompt for training."""
    ev_key = "prosecutor_evidence" if role == AgentRole.PROSECUTOR else "defense_evidence"
    private_ev = case.get(ev_key, [])
    priv_str = "\n".join(
        f"  - [{e.evidence_id}] {e.title}: {e.description}" for e in private_ev
    ) or "  (none)"

    pub_ev = public_evidence or []
    pub_str = "\n".join(
        f"  - [{e.evidence_id}] {e.title}: {e.description}" for e in pub_ev
    ) or "  (none revealed yet)"

    transcript = transcript or []
    recent = transcript[-4:]
    trans_str = "\n".join(
        f"  [{t.role.value.upper()} | {t.action_type.value}]: {t.argument[:200]}"
        for t in recent
    ) or "  (no arguments yet)"

    valid_actions = PHASE_ACTIONS.get(phase, ["ARGUE"])

    return f"""=== COURTROOM STATE ===
CASE: {case['case_id']}
BRIEF: {case['case_brief']}

YOUR ROLE: {role.value.upper()}
PHASE: {phase.value}
TURN: {turn}
VALID ACTIONS: {', '.join(valid_actions)}

YOUR PRIVATE EVIDENCE (only you can see this):
{priv_str}

PUBLIC EVIDENCE (revealed to all):
{pub_str}

RECENT TRANSCRIPT:
{trans_str}

SYSTEM: Your turn, {role.value}. Respond with valid XML."""


# ═══════════════════════════════════════════════════════════════════════════
#  3. RESPONSE PARSER (XML → VerdictAction)
# ═══════════════════════════════════════════════════════════════════════════

def parse_xml_response(text: str, phase: TrialPhase) -> VerdictAction:
    """Parse model output XML into a VerdictAction."""
    thinking = ""
    m = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    if m:
        thinking = m.group(1).strip()

    action_str = "ARGUE"
    m = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE)
    if m:
        action_str = m.group(1).strip().upper()

    argument = ""
    m = re.search(r"<argument>(.*?)</argument>", text, re.DOTALL)
    if m:
        argument = m.group(1).strip()

    evidence_id = None
    m = re.search(r"<evidence_id>(.*?)</evidence_id>", text, re.IGNORECASE)
    if m:
        evidence_id = m.group(1).strip()

    # Map to ActionType
    action_map = {
        "PLEA": ActionType.PLEA,
        "ARGUE": ActionType.ARGUE,
        "OBJECT": ActionType.OBJECT,
        "REVEAL_EVIDENCE": ActionType.REVEAL_EVIDENCE,
        "CONCEDE": ActionType.CONCEDE,
        "CLOSE": ActionType.CLOSE,
    }
    action_type = action_map.get(action_str, ActionType.ARGUE)

    # Truncate to 200 words
    words = argument.split()
    if len(words) > 200:
        argument = " ".join(words[:200])

    return VerdictAction(
        thinking=thinking or "No reasoning provided.",
        action_type=action_type,
        argument=argument or "No argument provided.",
        evidence_id=evidence_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. REWARD FUNCTIONS (for GRPO)
# ═══════════════════════════════════════════════════════════════════════════

def compute_format_reward(completion: str, **kwargs) -> float:
    """Reward for correct XML formatting (anti-gaming)."""
    score = 0.0
    if "<thinking>" in completion and "</thinking>" in completion:
        score += 0.3
    if "<action>" in completion and "</action>" in completion:
        score += 0.3
    if "<argument>" in completion and "</argument>" in completion:
        score += 0.4

    # Penalty for excessive length
    m = re.search(r"<argument>(.*?)</argument>", completion, re.DOTALL)
    if m:
        wc = len(m.group(1).split())
        if wc > 300:
            score -= 0.5
    return max(-1.0, score)


def compute_rubric_reward(completion: str, **kwargs) -> float:
    """Full rubric reward using VerdictEnvironment scoring."""
    case_id = kwargs.get("case_id", "")
    role_str = kwargs.get("role", "prosecutor")
    phase_str = kwargs.get("phase", "argument_rounds")

    try:
        phase = TrialPhase(phase_str)
        role = AgentRole(role_str)
    except ValueError:
        return 0.0

    action = parse_xml_response(completion, phase)

    # Build a temporary environment to score
    env = VerdictEnvironment(max_rounds=2)
    cases = load_cases()
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if not case and cases:
        case = cases[0]
    elif not case:
        return 0.0

    obs = env.reset(case)

    # Fast-forward to the target phase
    _advance_to_phase(env, phase, role)

    try:
        result = env.step(action)
        return result.reward
    except Exception:
        return 0.0


def _advance_to_phase(env: VerdictEnvironment, target: TrialPhase, target_role: AgentRole):
    """Advance environment to a target phase using filler actions."""
    filler = VerdictAction(
        thinking="Advancing to target phase.",
        action_type=ActionType.ARGUE,
        argument="The evidence presented supports our position and we request the court proceed.",
    )
    close = VerdictAction(
        thinking="Closing.",
        action_type=ActionType.CLOSE,
        argument="We rest our case and ask the court to consider the evidence.",
    )

    max_steps = 20
    for _ in range(max_steps):
        if env.state.phase == target and env.state.current_speaker == target_role:
            break
        if env.state.is_done:
            break
        act = close if env.state.phase == TrialPhase.CLOSING_STATEMENTS else filler
        env.step(act)


def compute_length_reward(completion: str, **kwargs) -> float:
    """Reward for appropriate argument length (20-200 words)."""
    m = re.search(r"<argument>(.*?)</argument>", completion, re.DOTALL)
    if not m:
        return -0.5
    wc = len(m.group(1).split())
    if 20 <= wc <= 200:
        return 0.5
    elif wc < 20:
        return max(-0.5, (wc / 20.0) - 0.5)
    else:
        return max(-0.5, 0.5 - (wc - 200) / 200.0)


# ═══════════════════════════════════════════════════════════════════════════
#  5. DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_training_dataset(difficulty: str | None = None, max_samples: int = 500):
    """Generate prompts by sampling case states at various phases."""
    from datasets import Dataset

    cases = load_cases(difficulty)
    if not cases:
        raise ValueError(f"No cases found for difficulty={difficulty}")

    roles = [AgentRole.PROSECUTOR, AgentRole.DEFENSE]
    phases = [
        TrialPhase.PLEA_BARGAIN,
        TrialPhase.OPENING_STATEMENTS,
        TrialPhase.ARGUMENT_ROUNDS,
        TrialPhase.CLOSING_STATEMENTS,
    ]

    records = []
    for case in cases:
        for role in roles:
            for phase in phases:
                # Generate a few variants with different transcript contexts
                for variant in range(2):
                    transcript = _generate_synthetic_transcript(case, role, phase, variant)
                    prompt = build_observation_prompt(
                        case=case,
                        role=role,
                        phase=phase,
                        turn=len(transcript),
                        transcript=transcript,
                    )

                    records.append({
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "case_id": case["case_id"],
                        "role": role.value,
                        "phase": phase.value,
                        "difficulty": case.get("difficulty", "medium"),
                    })

                    if len(records) >= max_samples:
                        break
                if len(records) >= max_samples:
                    break
            if len(records) >= max_samples:
                break
        if len(records) >= max_samples:
            break

    random.shuffle(records)
    print(f"📊 Built dataset: {len(records)} samples")
    return Dataset.from_list(records)


def _generate_synthetic_transcript(
    case: dict,
    current_role: AgentRole,
    target_phase: TrialPhase,
    variant: int = 0,
) -> list[TranscriptEntry]:
    """Generate a synthetic partial transcript up to the target phase."""
    transcript = []

    filler_args = [
        "The evidence presented clearly establishes the facts of this case and supports our position.",
        "We maintain our stance based on the documented record and ask the court to consider the timeline.",
        "The opposing counsel has failed to address the central issue. The facts speak for themselves.",
        "Upon examining the evidence, a clear pattern emerges that supports our argument.",
    ]

    # Build transcript entries up to the target phase
    phases_order = [
        TrialPhase.PLEA_BARGAIN,
        TrialPhase.OPENING_STATEMENTS,
        TrialPhase.ARGUMENT_ROUNDS,
    ]

    for phase in phases_order:
        if phase == target_phase:
            # Add opponent's entry if it's not prosecutor's first turn
            if current_role == AgentRole.DEFENSE:
                transcript.append(TranscriptEntry(
                    role=AgentRole.PROSECUTOR,
                    action_type=ActionType.ARGUE,
                    argument=random.choice(filler_args),
                    phase=phase,
                ))
            break

        # Both sides acted in this phase
        for role in [AgentRole.PROSECUTOR, AgentRole.DEFENSE]:
            transcript.append(TranscriptEntry(
                role=role,
                action_type=ActionType.ARGUE,
                argument=random.choice(filler_args),
                phase=phase,
            ))

    return transcript


# ═══════════════════════════════════════════════════════════════════════════
#  6. MODEL SETUP (Unsloth + LoRA)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, max_seq_length: int = 2048):
    """Load model with Unsloth 4-bit quantization + LoRA."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,  # auto-detect
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"✅ Model loaded with Unsloth: {model_name}")
        return model, tokenizer

    except ImportError:
        print("⚠️  Unsloth not found. Falling back to standard transformers.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        print(f"✅ Model loaded with transformers: {model_name}")
        return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  7. TRAINING LOOP (TRL GRPOTrainer)
# ═══════════════════════════════════════════════════════════════════════════

def train(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    difficulty: str | None = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    num_generations: int = 4,
    learning_rate: float = 5e-6,
    max_samples: int = 500,
    output_dir: str = "verdict-grpo-output",
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """Run GRPO training on the Verdict environment."""
    from trl import GRPOConfig, GRPOTrainer

    print("=" * 70)
    print("⚖️  VERDICT — GRPO Training")
    print("=" * 70)
    print(f"  Model:       {model_name}")
    print(f"  Difficulty:  {difficulty or 'all'}")
    print(f"  Epochs:      {num_epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Group size:  {num_generations}")
    print(f"  LR:          {learning_rate}")
    print(f"  Output:      {output_dir}")
    print()

    # Load model
    model, tokenizer = load_model(model_name)

    # Build dataset
    dataset = build_training_dataset(difficulty=difficulty, max_samples=max_samples)

    # GRPO Config
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        num_generations=num_generations,
        max_completion_length=512,
        max_prompt_length=1536,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_accumulation_steps=2,
        report_to="none",
        seed=42,
    )

    # Reward functions — composable, each weighted
    reward_funcs = [
        compute_format_reward,
        compute_rubric_reward,
        compute_length_reward,
    ]

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
    )

    print("🚀 Starting GRPO training...")
    trainer.train()

    # Save
    print(f"\n💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub and hub_model_id:
        print(f"📤 Pushing to HuggingFace Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)

    print("\n✅ Training complete!")
    return trainer


# ═══════════════════════════════════════════════════════════════════════════
#  8. CURRICULUM TRAINING (Easy → Medium → Hard)
# ═══════════════════════════════════════════════════════════════════════════

def train_curriculum(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    epochs_per_tier: int = 2,
    **kwargs,
):
    """Train with curriculum escalation: Easy → Medium → Hard."""
    print("=" * 70)
    print("⚖️  VERDICT — Curriculum GRPO Training")
    print("=" * 70)

    tiers = [
        ("easy", "Tier 1: Straightforward cases"),
        ("medium", "Tier 2: Conflicting evidence"),
        ("hard", "Tier 3: Asymmetric, adversarial cases"),
    ]

    for i, (diff, desc) in enumerate(tiers):
        print(f"\n{'─' * 70}")
        print(f"📈 {desc} (difficulty={diff})")
        print(f"{'─' * 70}")

        output_dir = f"verdict-grpo-tier{i + 1}-{diff}"

        # After first tier, load from previous checkpoint
        current_model = model_name if i == 0 else f"verdict-grpo-tier{i}-{tiers[i-1][0]}"

        train(
            model_name=current_model,
            difficulty=diff,
            num_epochs=epochs_per_tier,
            output_dir=output_dir,
            **kwargs,
        )

        model_name_for_next = output_dir

    print("\n🏆 Curriculum training complete!")
    print(f"   Final model: verdict-grpo-tier3-hard")


# ═══════════════════════════════════════════════════════════════════════════
#  9. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model_path: str, num_episodes: int = 5, difficulty: str | None = None):
    """Evaluate a trained model by running full courtroom episodes."""
    try:
        from transformers import pipeline
    except ImportError:
        print("❌ transformers not installed")
        return

    print(f"\n📊 Evaluating {model_path} on {num_episodes} episodes...")

    pipe = pipeline("text-generation", model=model_path, device_map="auto", torch_dtype="auto")
    cases = load_cases(difficulty)

    wins = {"prosecutor": 0, "defense": 0, "draw": 0}
    total_rewards = {"prosecutor": 0.0, "defense": 0.0}

    for ep in range(num_episodes):
        case = cases[ep % len(cases)]
        env = VerdictEnvironment(max_rounds=2)
        obs = env.reset(case)

        step = 0
        while not env.state.is_done and step < 20:
            role = env.state.current_speaker
            prompt = build_observation_prompt(
                case=case, role=role, phase=env.state.phase,
                turn=env.state.step_count, transcript=list(env.state.transcript),
                public_evidence=list(env.state.public_evidence),
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            outputs = pipe(messages, max_new_tokens=512, temperature=0.7,
                           do_sample=True, return_full_text=False)
            raw = outputs[0]["generated_text"]
            if isinstance(raw, list):
                raw = raw[-1].get("content", str(raw[-1]))

            action = parse_xml_response(str(raw), env.state.phase)
            result = env.step(action)
            total_rewards[role.value] += result.reward
            step += 1

        s = env.state
        if s.winner == AgentRole.PROSECUTOR:
            wins["prosecutor"] += 1
        elif s.winner == AgentRole.DEFENSE:
            wins["defense"] += 1
        else:
            wins["draw"] += 1

        print(f"  Episode {ep + 1}: Winner={s.winner.value if s.winner else 'DRAW'} "
              f"| Steps={s.step_count}")

    print(f"\n📊 Results over {num_episodes} episodes:")
    print(f"   Prosecutor wins: {wins['prosecutor']}")
    print(f"   Defense wins:    {wins['defense']}")
    print(f"   Draws:           {wins['draw']}")
    print(f"   Avg P reward:    {total_rewards['prosecutor'] / max(num_episodes, 1):.3f}")
    print(f"   Avg D reward:    {total_rewards['defense'] / max(num_episodes, 1):.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verdict GRPO Training")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    p_train = sub.add_parser("train", help="Run GRPO training")
    p_train.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_train.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"])
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--num-generations", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=5e-6)
    p_train.add_argument("--max-samples", type=int, default=500)
    p_train.add_argument("--output-dir", default="verdict-grpo-output")
    p_train.add_argument("--push-to-hub", action="store_true")
    p_train.add_argument("--hub-model-id", default=None)

    # Curriculum command
    p_curr = sub.add_parser("curriculum", help="Curriculum training (Easy→Med→Hard)")
    p_curr.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_curr.add_argument("--epochs-per-tier", type=int, default=2)
    p_curr.add_argument("--batch-size", type=int, default=4)

    # Eval command
    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("--model-path", required=True)
    p_eval.add_argument("--episodes", type=int, default=5)
    p_eval.add_argument("--difficulty", default=None)

    args = parser.parse_args()

    if args.command == "train":
        train(
            model_name=args.model,
            difficulty=args.difficulty,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            num_generations=args.num_generations,
            learning_rate=args.lr,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )
    elif args.command == "curriculum":
        train_curriculum(
            model_name=args.model,
            epochs_per_tier=args.epochs_per_tier,
            batch_size=args.batch_size,
        )
    elif args.command == "eval":
        evaluate_model(
            model_path=args.model_path,
            num_episodes=args.episodes,
            difficulty=args.difficulty,
        )
    else:
        parser.print_help()
