"""
Verdict LLM Agent — Wires an LLM into the courtroom environment.
=================================================================
The agent reads VerdictObservation, constructs a prompt, calls the LLM,
parses the response into a VerdictAction, and feeds it back to env.step().

Supports: HuggingFace Transformers (local) or any OpenAI-compatible API.
"""

from __future__ import annotations

import json
import re
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from models import (
    ActionType, AgentRole, TrialPhase,
    VerdictAction, VerdictObservation,
)

# ---------------------------------------------------------------------------
#  Prompt Builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a courtroom agent in a legal trial simulation.
You will receive a case brief, your role (Prosecutor or Defense), the trial phase,
the transcript so far, your private evidence, and public evidence.

You must respond with VALID JSON matching this exact schema:
{
  "thinking": "<your internal reasoning — theory of mind, strategy, what you think the opponent will do>",
  "action_type": "<one of: plea, argue, object, reveal_evidence, concede, close>",
  "argument": "<your spoken statement to the court, max 200 words>",
  "evidence_id": "<optional: ID of evidence to reveal, e.g. P1 or D1>"
}

RULES:
- During plea_bargain: use "plea" to accept a plea deal, or "argue" to reject and proceed to trial.
- During opening_statements: use "argue" only.
- During argument_rounds: use "argue", "object", "reveal_evidence", or "concede".
- During closing_statements: use "close" only.
- When using "reveal_evidence", you MUST include "evidence_id" matching one of your private evidence cards.
- Your "thinking" field is private (the opponent cannot see it). Use it for strategy.
- Your "argument" field is public (everyone in the courtroom sees it).
- Reference evidence by title in your argument for maximum score.
- Directly address the opponent's last argument for counter_quality points.
- Do NOT repeat yourself across turns (consistency penalty).
"""


def build_user_prompt(obs: VerdictObservation) -> str:
    """Convert a VerdictObservation into a structured prompt for the LLM."""
    # Valid actions for this phase
    phase_actions = {
        TrialPhase.PLEA_BARGAIN: ["plea", "argue"],
        TrialPhase.OPENING_STATEMENTS: ["argue"],
        TrialPhase.ARGUMENT_ROUNDS: ["argue", "object", "reveal_evidence", "concede"],
        TrialPhase.CLOSING_STATEMENTS: ["close"],
    }
    valid = phase_actions.get(obs.phase, ["argue"])

    # Private evidence
    priv_ev = ""
    if obs.private_evidence:
        priv_ev = "\n".join(
            f"  - [{e.evidence_id}] {e.title}: {e.description}"
            for e in obs.private_evidence
        )
    else:
        priv_ev = "  (none remaining)"

    # Public evidence
    pub_ev = ""
    if obs.public_evidence:
        pub_ev = "\n".join(
            f"  - [{e.evidence_id}] {e.title}: {e.description}"
            for e in obs.public_evidence
        )
    else:
        pub_ev = "  (none revealed yet)"

    # Recent transcript (last 4 entries to save context window)
    transcript_lines = ""
    recent = obs.transcript[-4:] if obs.transcript else []
    if recent:
        transcript_lines = "\n".join(
            f"  [{t.role.value.upper()} | {t.action_type.value}]: {t.argument[:200]}"
            for t in recent
        )
    else:
        transcript_lines = "  (no arguments yet)"

    return f"""=== COURTROOM STATE ===
CASE: {obs.case_id}
BRIEF: {obs.case_brief}

YOUR ROLE: {obs.role.value.upper()}
PHASE: {obs.phase.value}
TURN: {obs.turn_number}
VALID ACTIONS: {', '.join(valid)}

YOUR PRIVATE EVIDENCE (only you can see this):
{priv_ev}

PUBLIC EVIDENCE (revealed to all):
{pub_ev}

RECENT TRANSCRIPT:
{transcript_lines}

SYSTEM: {obs.message}

Respond with valid JSON only. No markdown, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
#  Response Parser
# ---------------------------------------------------------------------------

def parse_llm_response(raw: str, obs: VerdictObservation) -> VerdictAction:
    """Parse LLM text output into a VerdictAction. Handles common LLM quirks."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            # Fallback: construct a default action
            return _fallback_action(obs)

    # Validate and construct
    thinking = str(data.get("thinking", "No reasoning provided."))
    action_type_str = str(data.get("action_type", "argue")).lower().strip()
    argument = str(data.get("argument", "No argument provided."))
    evidence_id = data.get("evidence_id")

    # Map action type
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        action_type = ActionType.ARGUE  # safe fallback

    # Truncate argument if too long
    words = argument.split()
    if len(words) > 200:
        argument = " ".join(words[:200])

    return VerdictAction(
        thinking=thinking,
        action_type=action_type,
        argument=argument,
        evidence_id=evidence_id,
    )


def _fallback_action(obs: VerdictObservation) -> VerdictAction:
    """Generate a safe fallback action when LLM output can't be parsed."""
    phase_default = {
        TrialPhase.PLEA_BARGAIN: ActionType.ARGUE,
        TrialPhase.OPENING_STATEMENTS: ActionType.ARGUE,
        TrialPhase.ARGUMENT_ROUNDS: ActionType.ARGUE,
        TrialPhase.CLOSING_STATEMENTS: ActionType.CLOSE,
    }
    action_type = phase_default.get(obs.phase, ActionType.ARGUE)
    return VerdictAction(
        thinking="[FALLBACK] LLM output could not be parsed.",
        action_type=action_type,
        argument="The evidence speaks for itself. We maintain our position.",
        evidence_id=None,
    )


# ---------------------------------------------------------------------------
#  LLM Agent (HuggingFace Transformers — local inference)
# ---------------------------------------------------------------------------

class VerdictLLMAgent:
    """Wraps a HuggingFace model to play one side of the courtroom.

    Usage:
        agent = VerdictLLMAgent(model_name="Qwen/Qwen2.5-1.5B-Instruct")
        obs = env.reset()
        action = agent.act(obs)
        result = env.step(action)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = None
        self._device = device

    def _load(self):
        """Lazy-load the model pipeline."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map=self._device,
                torch_dtype="auto",
            )
            print(f"✅ Loaded model: {self.model_name}")
        except Exception as e:
            print(f"⚠️  Failed to load {self.model_name}: {e}")
            print("   Falling back to rule-based agent.")
            self._pipeline = "FALLBACK"

    def act(self, obs: VerdictObservation) -> VerdictAction:
        """Given an observation, generate an action via the LLM."""
        self._load()

        if self._pipeline == "FALLBACK":
            return _fallback_action(obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(obs)},
        ]

        try:
            outputs = self._pipeline(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                return_full_text=False,
            )
            raw = outputs[0]["generated_text"]
            if isinstance(raw, list):
                raw = raw[-1].get("content", str(raw[-1]))
            return parse_llm_response(str(raw), obs)
        except Exception as e:
            print(f"⚠️  LLM generation error: {e}")
            return _fallback_action(obs)


# ---------------------------------------------------------------------------
#  Run a full LLM-vs-LLM episode
# ---------------------------------------------------------------------------

def run_llm_episode(
    prosecutor_agent: VerdictLLMAgent,
    defense_agent: VerdictLLMAgent,
    max_rounds: int = 2,
    verbose: bool = True,
):
    """Run one full courtroom episode with two LLM agents."""
    from verdict_environment import VerdictEnvironment

    env = VerdictEnvironment(max_rounds=max_rounds)
    obs = env.reset()

    if verbose:
        s = env.state
        print(f"📋 Case: {s.charge}")
        print(f"   Brief: {s.case_brief[:100]}...")
        print()

    step = 0
    rewards = {"prosecutor": 0.0, "defense": 0.0}

    while not env.state.is_done:
        role = env.state.current_speaker
        agent = prosecutor_agent if role == AgentRole.PROSECUTOR else defense_agent

        action = agent.act(obs)
        result = env.step(action)

        rewards[role.value] += result.reward
        step += 1

        if verbose:
            print(f"  Step {step:2d} | {role.value.upper():10s} | "
                  f"{action.action_type.value:16s} | reward={result.reward:.3f} | "
                  f"phase={env.state.phase.value}")
            if action.action_type == ActionType.REVEAL_EVIDENCE:
                print(f"         → Revealed: {action.evidence_id}")

        obs = result.observation

        # Safety: prevent infinite loops
        if step > 30:
            print("⚠️  Safety limit: breaking at 30 steps.")
            break

    if verbose:
        s = env.state
        print(f"\n⚖️  VERDICT: {s.verdict}")
        print(f"   Winner: {s.winner.value if s.winner else 'DRAW'}")
        print(f"   Prosecutor reward: {rewards['prosecutor']:.3f}")
        print(f"   Defense reward:    {rewards['defense']:.3f}")

    return env.state, rewards


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Verdict with LLM agents")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model name")
    parser.add_argument("--rounds", type=int, default=2, help="Argument rounds per episode")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    args = parser.parse_args()

    print("=" * 60)
    print("VERDICT — LLM Agent Episode")
    print("=" * 60)

    prosecutor = VerdictLLMAgent(model_name=args.model, device=args.device)
    defense = VerdictLLMAgent(model_name=args.model, device=args.device)

    state, rewards = run_llm_episode(prosecutor, defense, max_rounds=args.rounds)
    print("\n✅ Done.")
