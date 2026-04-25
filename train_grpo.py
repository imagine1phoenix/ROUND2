"""
Verdict GRPO training script.

This script is built against the files that are present in this repository:

- server/verdict_environment.py for rollout and reward scoring
- server/models.py for VerdictAction and phase/role enums
- client/verdict_agent.py for the JSON prompt contract and parser
- cases.json through verdict_environment.load_cases

Common usage:
    python3 train_grpo.py smoke --samples 8
    python3 train_grpo.py train --difficulty easy --max-samples 64
    python3 train_grpo.py eval --model-path verdict-grpo-output --episodes 5

Training dependencies are intentionally imported only inside the training path:
    pip install torch transformers datasets trl peft accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Any, Iterable


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(ROOT_DIR, "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from client.verdict_agent import SYSTEM_PROMPT, build_user_prompt, parse_llm_response
from models import ActionType, AgentRole, TrialPhase, VerdictAction, VerdictObservation
from verdict_environment import VerdictEnvironment, load_cases


PHASES_FOR_TRAINING = (
    TrialPhase.PLEA_BARGAIN,
    TrialPhase.OPENING_STATEMENTS,
    TrialPhase.ARGUMENT_ROUNDS,
    TrialPhase.CLOSING_STATEMENTS,
)

ROLES_FOR_TRAINING = (AgentRole.PROSECUTOR, AgentRole.DEFENSE)

VALID_ACTIONS_BY_PHASE = {
    TrialPhase.PLEA_BARGAIN: {ActionType.PLEA, ActionType.ARGUE},
    TrialPhase.OPENING_STATEMENTS: {ActionType.ARGUE},
    TrialPhase.ARGUMENT_ROUNDS: {
        ActionType.ARGUE,
        ActionType.OBJECT,
        ActionType.REVEAL_EVIDENCE,
        ActionType.CONCEDE,
    },
    TrialPhase.CLOSING_STATEMENTS: {ActionType.CLOSE},
}


def current_observation(env: VerdictEnvironment) -> VerdictObservation:
    """Return an observation for the actual current speaker.

    The environment currently returns the previous speaker's observation from
    step(), so the training script asks the environment for the current speaker
    explicitly to avoid leaking the wrong private evidence into prompts.
    """
    return env._get_observation(env.state.current_speaker)


def filler_action(role: AgentRole, phase: TrialPhase) -> VerdictAction:
    """Deterministic action used to advance the environment to sampled states."""
    side = "prosecution" if role == AgentRole.PROSECUTOR else "defense"

    if phase == TrialPhase.CLOSING_STATEMENTS:
        return VerdictAction(
            thinking=f"The {side} should close by tying the record to its theory.",
            action_type=ActionType.CLOSE,
            argument=(
                f"The {side} rests on the documented record, the timeline of events, "
                "and the burden the court must apply. The opposing narrative leaves "
                "important gaps unresolved, so the court should weigh the evidence "
                "carefully and rule in our favor."
            ),
        )

    return VerdictAction(
        thinking=f"The {side} should preserve a clear position and address the record.",
        action_type=ActionType.ARGUE,
        argument=(
            f"The {side} rejects an unsupported reading of the facts. The case record "
            "shows a coherent timeline, and the opposing side has not explained the "
            "most important evidence. This court should focus on the documented facts "
            "rather than speculation."
        ),
    )


def prepare_env_at_state(
    case_data: dict[str, Any],
    target_phase: TrialPhase,
    target_role: AgentRole,
    max_rounds: int,
) -> VerdictEnvironment:
    """Create an environment and advance it to a reachable phase/role pair."""
    env = VerdictEnvironment(max_rounds=max_rounds)
    env.reset(case_data)

    for _ in range(40):
        if env.state.phase == target_phase and env.state.current_speaker == target_role:
            return env
        if env.state.is_done:
            break
        env.step(filler_action(env.state.current_speaker, env.state.phase))

    raise RuntimeError(
        f"Could not reach phase={target_phase.value} role={target_role.value} "
        f"for case={case_data.get('case_id')}"
    )


def build_prompt_messages(obs: VerdictObservation) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(obs)},
    ]


def build_training_records(
    difficulty: str | None,
    max_samples: int,
    max_rounds: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate GRPO prompt records from real Verdict cases and states."""
    random.seed(seed)
    cases = load_cases(difficulty)
    if not cases:
        raise ValueError(f"No cases found for difficulty={difficulty!r}")

    records: list[dict[str, Any]] = []
    for case in cases:
        for phase in PHASES_FOR_TRAINING:
            for role in ROLES_FOR_TRAINING:
                env = prepare_env_at_state(case, phase, role, max_rounds=max_rounds)
                obs = current_observation(env)
                records.append(
                    {
                        "prompt": build_prompt_messages(obs),
                        "case_id": case["case_id"],
                        "role": role.value,
                        "phase": phase.value,
                        "difficulty": case.get("difficulty", "medium"),
                        "max_rounds": max_rounds,
                    }
                )

    random.shuffle(records)
    return records[:max_samples]


def completion_to_text(completion: Any) -> str:
    """Handle both standard-text and conversational TRL completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output without accepting markdown prose."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def strict_action_from_completion(
    completion: Any,
    obs: VerdictObservation,
) -> VerdictAction | None:
    """Parse completion as a VerdictAction, rejecting malformed JSON."""
    text = completion_to_text(completion)
    data = extract_json_object(text)
    if data is None:
        return None

    required = {"thinking", "action_type", "argument"}
    if not required.issubset(data):
        return None

    try:
        return parse_llm_response(json.dumps(data), obs)
    except Exception:
        return None


def action_is_valid_for_phase(action: VerdictAction, phase: TrialPhase) -> bool:
    return action.action_type in VALID_ACTIONS_BY_PHASE.get(phase, {ActionType.ARGUE})


def find_case(case_id: str, difficulty: str | None = None) -> dict[str, Any]:
    cases = load_cases(difficulty)
    for case in cases:
        if case["case_id"] == case_id:
            return case
    all_cases = load_cases()
    for case in all_cases:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"Unknown case_id={case_id}")


def score_one_completion(
    completion: Any,
    case_id: str,
    role: str,
    phase: str,
    max_rounds: int,
) -> float:
    """Score one completion through the real VerdictEnvironment rubric."""
    target_role = AgentRole(role)
    target_phase = TrialPhase(phase)
    case = find_case(case_id)
    env = prepare_env_at_state(case, target_phase, target_role, max_rounds=max_rounds)
    obs = current_observation(env)

    action = strict_action_from_completion(completion, obs)
    if action is None:
        return -0.5
    if not action_is_valid_for_phase(action, target_phase):
        return -0.25
    if action.action_type == ActionType.REVEAL_EVIDENCE and not action.evidence_id:
        return -0.25

    try:
        return float(env.step(action).reward)
    except Exception:
        return -0.5


def format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Small reward for strict JSON matching the live VerdictAction contract."""
    rewards: list[float] = []
    for completion in completions:
        data = extract_json_object(completion_to_text(completion))
        if data is None:
            rewards.append(-0.4)
            continue

        score = 0.0
        score += 0.15 if isinstance(data.get("thinking"), str) and data["thinking"].strip() else -0.1
        score += 0.25 if data.get("action_type") in {a.value for a in ActionType} else -0.2
        argument = data.get("argument")
        if isinstance(argument, str) and argument.strip():
            word_count = len(argument.split())
            score += 0.25
            if 20 <= word_count <= 200:
                score += 0.2
            elif word_count > 300:
                score -= 0.2
        else:
            score -= 0.2
        rewards.append(score)
    return rewards


def phase_action_reward(
    completions: list[Any],
    phase: list[str],
    case_id: list[str],
    role: list[str],
    max_rounds: list[int],
    **kwargs: Any,
) -> list[float]:
    """Reward actions that are legal in the prompted phase."""
    rewards: list[float] = []
    for completion, case, side, ph, rounds in zip(completions, case_id, role, phase, max_rounds):
        env = prepare_env_at_state(find_case(case), TrialPhase(ph), AgentRole(side), max_rounds=int(rounds))
        action = strict_action_from_completion(completion, current_observation(env))
        if action is None:
            rewards.append(-0.2)
        elif action_is_valid_for_phase(action, TrialPhase(ph)):
            rewards.append(0.2)
        else:
            rewards.append(-0.3)
    return rewards


def verdict_environment_reward(
    completions: list[Any],
    case_id: list[str],
    role: list[str],
    phase: list[str],
    max_rounds: list[int],
    **kwargs: Any,
) -> list[float]:
    """Main dense reward from VerdictEnvironment._compute_rubric."""
    return [
        score_one_completion(completion, case, side, ph, int(rounds))
        for completion, case, side, ph, rounds in zip(completions, case_id, role, phase, max_rounds)
    ]


def make_dataset(records: list[dict[str, Any]]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for training. Install with: "
            "pip install datasets"
        ) from exc
    return Dataset.from_list(records)


def train_grpo(args: argparse.Namespace) -> None:
    try:
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: "
            "pip install torch transformers datasets trl peft accelerate"
        ) from exc

    records = build_training_records(
        difficulty=args.difficulty,
        max_samples=args.max_samples,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )
    dataset = make_dataset(records)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    peft_config = None
    if args.use_peft:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=[format_reward, phase_action_reward, verdict_environment_reward],
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


def generate_text(pipe: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
    )
    generated = outputs[0]["generated_text"]
    if isinstance(generated, list) and generated:
        return str(generated[-1].get("content", generated[-1]))
    return str(generated)


def evaluate_model(args: argparse.Namespace) -> None:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers to run evaluation.") from exc

    cases = load_cases(args.difficulty)
    if not cases:
        raise ValueError(f"No cases found for difficulty={args.difficulty!r}")

    pipe = pipeline("text-generation", model=args.model_path, device_map=args.device, torch_dtype="auto")
    wins = {"prosecutor": 0, "defense": 0, "draw": 0}
    rewards = {"prosecutor": 0.0, "defense": 0.0}

    for episode in range(args.episodes):
        case = cases[episode % len(cases)]
        env = VerdictEnvironment(max_rounds=args.max_rounds)
        env.reset(case)

        for _ in range(args.max_steps):
            if env.state.is_done:
                break
            obs = current_observation(env)
            raw = generate_text(pipe, build_prompt_messages(obs), args.max_completion_length)
            action = parse_llm_response(raw, obs)
            role = env.state.current_speaker.value
            result = env.step(action)
            rewards[role] += result.reward

        winner = env.state.winner.value if env.state.winner else "draw"
        wins[winner] += 1
        print(
            f"episode={episode + 1} case={case['case_id']} "
            f"winner={winner} steps={env.state.step_count} verdict={env.state.verdict}"
        )

    print("\nEvaluation summary")
    print(f"prosecutor_wins={wins['prosecutor']}")
    print(f"defense_wins={wins['defense']}")
    print(f"draws={wins['draw']}")
    print(f"avg_prosecutor_reward={rewards['prosecutor'] / max(args.episodes, 1):.3f}")
    print(f"avg_defense_reward={rewards['defense'] / max(args.episodes, 1):.3f}")


def run_smoke(args: argparse.Namespace) -> None:
    records = build_training_records(
        difficulty=args.difficulty,
        max_samples=args.samples,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )
    print(f"records={len(records)}")
    if not records:
        return

    first = records[0]
    print(json.dumps({k: v for k, v in first.items() if k != "prompt"}, indent=2))
    print("\nPrompt preview")
    for message in first["prompt"]:
        print(f"[{message['role']}]\n{message['content'][:1200]}\n")

    sample_completion = json.dumps(
        {
            "thinking": "I should address the opposing timeline and keep the claim grounded.",
            "action_type": "argue",
            "argument": (
                "The documented timeline supports our side because it connects the key facts "
                "to a concrete legal theory. The opposing account leaves important gaps and "
                "does not explain why the evidence should be read differently. The court "
                "should therefore focus on the record rather than speculation."
            ),
            "evidence_id": None,
        }
    )
    reward = score_one_completion(
        sample_completion,
        case_id=first["case_id"],
        role=first["role"],
        phase=first["phase"],
        max_rounds=int(first["max_rounds"]),
    )
    print(f"sample_reward={reward:.3f}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Verdict agents with TRL GRPO.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Build prompts and score one sample without training deps.")
    add_common_args(smoke)
    smoke.add_argument("--samples", type=int, default=8)
    smoke.set_defaults(func=run_smoke)

    train = subparsers.add_parser("train", help="Run GRPO training.")
    add_common_args(train)
    train.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    train.add_argument("--output-dir", default="verdict-grpo-output")
    train.add_argument("--max-samples", type=int, default=240)
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--num-generations", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=5e-6)
    train.add_argument("--max-prompt-length", type=int, default=1536)
    train.add_argument("--max-completion-length", type=int, default=384)
    train.add_argument("--logging-steps", type=int, default=5)
    train.add_argument("--save-steps", type=int, default=50)
    train.add_argument("--use-peft", action="store_true")
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.set_defaults(func=train_grpo)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model in full episodes.")
    add_common_args(eval_parser)
    eval_parser.add_argument("--model-path", required=True)
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--max-steps", type=int, default=30)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--max-completion-length", type=int, default=384)
    eval_parser.set_defaults(func=evaluate_model)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
