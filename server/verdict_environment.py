"""
Verdict Environment — Core RL Logic
=====================================
OpenEnv-compliant courtroom simulation.
Subclasses Environment with reset(), step(), and state.
"""

from __future__ import annotations

import os
import random
import re
import uuid
from typing import Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server import Environment as OpenEnvBase
except ImportError:
    OpenEnvBase = object  # local dev fallback

from models import (
    ActionType, AgentRole, TrialPhase,
    EvidenceCard, TranscriptEntry, RubricScore,
    VerdictAction, VerdictObservation, VerdictState,
)

# ---------------------------------------------------------------------------
#  Case Bank Loader (reads cases.json, supports difficulty filtering)
# ---------------------------------------------------------------------------

_CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "cases.json")


def _evidence_from_strings(items: list[str], prefix: str) -> list[EvidenceCard]:
    """Convert evidence string list from cases.json into EvidenceCard objects."""
    return [
        EvidenceCard(
            evidence_id=f"{prefix}{i+1}",
            title=text.split(" showing ")[0].split(" confirming ")[0].split(" citing ")[0][:60],
            description=text,
            revealed=False,
        )
        for i, text in enumerate(items)
    ]


def load_cases(
    path: str = _CASES_PATH,
    difficulty: Optional[str] = None,
) -> list[dict]:
    """Load cases from JSON file and optionally filter by difficulty tier.

    Args:
        path: Path to cases.json
        difficulty: Filter by 'easy', 'medium', or 'hard'. None = all cases.

    Returns:
        List of case dicts with EvidenceCard objects ready for the environment.
    """
    import json as _json

    with open(path, "r") as f:
        raw_cases = _json.load(f)

    if difficulty:
        raw_cases = [c for c in raw_cases if c.get("difficulty") == difficulty]

    cases = []
    for c in raw_cases:
        cases.append({
            "case_id": c["id"],
            "charge": c["charge"],
            "case_brief": c["facts"],
            "difficulty": c.get("difficulty", "medium"),
            "category": c.get("category", ""),
            "title": c.get("title", ""),
            "prosecutor_evidence": _evidence_from_strings(c["prosecutor_evidence"], "P"),
            "defense_evidence": _evidence_from_strings(c["defense_evidence"], "D"),
        })
    return cases


# Pre-load all cases at module level (fallback to inline if file missing)
try:
    CASE_BANK = load_cases()
except FileNotFoundError:
    CASE_BANK = []

# ---------------------------------------------------------------------------
#  Heuristic Scoring Functions (Composable Rubric)
# ---------------------------------------------------------------------------

LOGICAL_CONNECTORS = {
    "therefore", "because", "however", "furthermore", "consequently",
    "moreover", "nevertheless", "thus", "hence", "whereas", "although",
    "despite", "in contrast", "on the contrary", "as a result",
}

REBUTTAL_MARKERS = {
    "however", "on the contrary", "this fails", "this ignores",
    "the prosecution fails", "the defense fails", "incorrect",
    "misrepresents", "overlooks", "contradicts", "flawed",
    "but", "yet", "nonetheless", "refute", "counter", "rebut",
    "objection", "misleading", "unfounded", "baseless",
}


def score_coherence(argument: str, case_brief: str) -> float:
    """30% weight — logical structure and case-fact grounding."""
    if not argument.strip():
        return 0.0
    score = 0.0
    words = argument.lower().split()
    # Sentence structure (>2 sentences = structured argument)
    sentences = [s.strip() for s in re.split(r'[.!?]', argument) if s.strip()]
    if len(sentences) >= 2:
        score += 0.3
    # Logical connectors
    connector_count = sum(1 for c in LOGICAL_CONNECTORS if c in argument.lower())
    score += min(connector_count * 0.15, 0.3)
    # References case facts (keyword overlap with case brief)
    brief_words = set(case_brief.lower().split())
    overlap = len(set(words) & brief_words) / max(len(brief_words), 1)
    score += min(overlap * 2.0, 0.4)
    return min(score, 1.0)


def score_evidence_usage(
    argument: str,
    action_type: ActionType,
    evidence_id: Optional[str],
    available_evidence: list[EvidenceCard],
    public_evidence: list[EvidenceCard],
) -> float:
    """20% weight — strategic and correct use of evidence."""
    score = 0.0
    all_ev = available_evidence + public_evidence
    ev_titles = [e.title.lower() for e in all_ev]
    ev_ids = [e.evidence_id.lower() for e in all_ev]
    arg_lower = argument.lower()
    # Did they reference any evidence by title or ID?
    refs = sum(1 for t in ev_titles if t in arg_lower)
    refs += sum(1 for eid in ev_ids if eid in arg_lower)
    if refs > 0:
        score += 0.5
    # Strategic reveal
    if action_type == ActionType.REVEAL_EVIDENCE and evidence_id:
        valid_ids = [e.evidence_id for e in available_evidence if not e.revealed]
        if evidence_id in valid_ids:
            score += 0.3
        # Evidence dump penalty: reveal without narrative
        if len(argument.split()) < 15:
            score = max(score - 0.4, 0.0)
    # Contextual integration (evidence woven into argument)
    if refs > 0 and len(argument.split()) > 30:
        score += 0.2
    return min(score, 1.0)


def score_counter_quality(
    argument: str, transcript: list[TranscriptEntry], role: AgentRole
) -> float:
    """20% weight — did the agent address the opponent's last point?"""
    if not transcript:
        return 0.5  # no opponent move yet, neutral score
    # Find opponent's last argument
    opponent = AgentRole.DEFENSE if role == AgentRole.PROSECUTOR else AgentRole.PROSECUTOR
    opponent_args = [t for t in transcript if t.role == opponent]
    if not opponent_args:
        return 0.5
    last_opponent = opponent_args[-1].argument.lower()
    arg_lower = argument.lower()
    # Keyword overlap with opponent's last argument
    opp_words = set(last_opponent.split()) - {"the", "a", "an", "is", "was", "to", "of", "and", "in", "for"}
    overlap = len(set(arg_lower.split()) & opp_words)
    score = min(overlap * 0.1, 0.5)
    # Rebuttal language
    rebuttal_count = sum(1 for m in REBUTTAL_MARKERS if m in arg_lower)
    score += min(rebuttal_count * 0.15, 0.5)
    return min(score, 1.0)


def score_consistency(
    argument: str, transcript: list[TranscriptEntry], role: AgentRole
) -> float:
    """15% weight — no self-contradiction or repetition loops."""
    own_args = [t.argument.lower() for t in transcript if t.role == role]
    if not own_args:
        return 1.0  # first turn, fully consistent
    arg_lower = argument.lower()
    arg_words = set(arg_lower.split())
    # Repetition penalty: >60% overlap with any own previous argument
    for prev in own_args:
        prev_words = set(prev.split())
        if not prev_words:
            continue
        overlap = len(arg_words & prev_words) / max(len(prev_words), 1)
        if overlap > 0.6:
            return max(0.2, 1.0 - overlap)
    return 1.0


def compute_format_penalty(action: VerdictAction) -> float:
    """Rule-based format penalties from RULES.md."""
    penalty = 0.0
    if not action.thinking.strip():
        penalty -= 1.0
    if not action.argument.strip():
        penalty -= 1.0
    word_count = len(action.argument.split())
    if word_count > 300:
        penalty -= 1.0
    valid_actions = {a.value for a in ActionType}
    if action.action_type.value not in valid_actions:
        penalty -= 0.5
    return penalty


def compute_rubric(
    action: VerdictAction,
    state: VerdictState,
    role: AgentRole,
) -> tuple[RubricScore, float]:
    """Compute the full 5-dimension composable rubric + format penalty."""
    evidence = state.prosecutor_evidence if role == AgentRole.PROSECUTOR else state.defense_evidence
    rubric = RubricScore(
        coherence=score_coherence(action.argument, state.case_brief),
        evidence_usage=score_evidence_usage(
            action.argument, action.action_type, action.evidence_id,
            evidence, state.public_evidence,
        ),
        counter_quality=score_counter_quality(action.argument, state.transcript, role),
        consistency=score_consistency(action.argument, state.transcript, role),
        verdict_alignment=0.0,  # filled at terminal step
    )
    fmt_penalty = compute_format_penalty(action)
    reward = rubric.weighted_total + fmt_penalty
    return rubric, reward


# ---------------------------------------------------------------------------
#  Phase Validation
# ---------------------------------------------------------------------------

VALID_ACTIONS_PER_PHASE = {
    TrialPhase.PLEA_BARGAIN: {ActionType.PLEA, ActionType.ARGUE},
    TrialPhase.OPENING_STATEMENTS: {ActionType.ARGUE},
    TrialPhase.ARGUMENT_ROUNDS: {ActionType.ARGUE, ActionType.OBJECT, ActionType.REVEAL_EVIDENCE, ActionType.CONCEDE},
    TrialPhase.CLOSING_STATEMENTS: {ActionType.CLOSE},
}


# ---------------------------------------------------------------------------
#  Step Result
# ---------------------------------------------------------------------------

class VerdictStepResult(BaseModel):
    """Returned by step(). Mirrors OpenEnv StepResult contract."""
    observation: VerdictObservation
    reward: float = 0.0
    done: bool = False
    info: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
#  The Environment
# ---------------------------------------------------------------------------

class VerdictEnvironment(OpenEnvBase):
    """OpenEnv-compliant multi-agent courtroom simulation.

    API contract:
        reset()  → VerdictObservation   (initial obs for Prosecutor)
        step()   → VerdictStepResult    (obs, reward, done, info)
        state    → VerdictState          (full episode state)
    """

    def __init__(self, max_rounds: int = 4, difficulty: Optional[str] = None):
        if OpenEnvBase is not object:
            super().__init__()
        self._max_rounds = max_rounds
        self._difficulty = difficulty  # 'easy', 'medium', 'hard', or None (all)
        self._case_pool = self._load_case_pool()
        self._state = VerdictState()
        self._last_reward: float = 0.0

    def _load_case_pool(self) -> list[dict]:
        """Load cases filtered by difficulty tier."""
        if self._difficulty:
            pool = load_cases(difficulty=self._difficulty)
        else:
            pool = list(CASE_BANK)
        if not pool:
            pool = list(CASE_BANK)  # fallback to all if filter returns empty
        return pool

    # ---- Gym-style API ------------------------------------------------

    def reset(self, difficulty: Optional[str] = None) -> VerdictObservation:
        """Initialize a new episode with a random case.

        Args:
            difficulty: Override the instance-level difficulty for this episode.
                        'easy', 'medium', 'hard', or None (use instance default).
        """
        # Allow per-episode difficulty override for curriculum training
        if difficulty and difficulty != self._difficulty:
            pool = load_cases(difficulty=difficulty)
            if not pool:
                pool = self._case_pool
        else:
            pool = self._case_pool

        case = random.choice(pool)
        self._state = VerdictState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            case_id=case["case_id"],
            case_brief=case["case_brief"],
            charge=case["charge"],
            phase=TrialPhase.PLEA_BARGAIN,
            current_speaker=AgentRole.PROSECUTOR,
            round_number=0,
            max_rounds=self._max_rounds,
            prosecutor_evidence=[e.model_copy() for e in case["prosecutor_evidence"]],
            defense_evidence=[e.model_copy() for e in case["defense_evidence"]],
            public_evidence=[],
            transcript=[],
            prosecutor_scores=[],
            defense_scores=[],
            prosecutor_plea=None,
            defense_plea=None,
            is_done=False,
            verdict=None,
            winner=None,
        )
        self._last_reward = 0.0
        return self._build_observation(
            role=AgentRole.PROSECUTOR,
            message=f"Trial begins. Case: {case['charge']}. Prosecutor, you may propose a plea or proceed to trial.",
        )

    def step(self, action: VerdictAction) -> VerdictStepResult:
        """Execute one agent action and advance the trial."""
        s = self._state
        role = s.current_speaker

        # --- 1. Validate action for current phase ---
        valid = VALID_ACTIONS_PER_PHASE.get(s.phase, set())
        if valid and action.action_type not in valid:
            penalty = -0.5
            self._last_reward = penalty
            return VerdictStepResult(
                observation=self._build_observation(role, f"Invalid action '{action.action_type.value}' during {s.phase.value}. Turn forfeited."),
                reward=penalty,
                done=False,
            )

        # --- 2. Handle evidence reveal ---
        if action.action_type == ActionType.REVEAL_EVIDENCE:
            self._handle_evidence_reveal(action, role)

        # --- 3. Record transcript ---
        entry = TranscriptEntry(
            turn_number=s.step_count,
            role=role,
            action_type=action.action_type,
            argument=action.argument,
            evidence_revealed=action.evidence_id if action.action_type == ActionType.REVEAL_EVIDENCE else None,
            phase=s.phase,
        )
        s.transcript.append(entry)
        s.step_count += 1

        # --- 4. Compute rubric reward ---
        rubric, reward = compute_rubric(action, s, role)
        if role == AgentRole.PROSECUTOR:
            s.prosecutor_scores.append(rubric)
        else:
            s.defense_scores.append(rubric)
        self._last_reward = reward

        # --- 5. Handle plea bargain ---
        plea_handled = False
        if s.phase == TrialPhase.PLEA_BARGAIN:
            done, msg = self._handle_plea(action, role)
            plea_handled = True
            if done:
                return VerdictStepResult(
                    observation=self._build_observation(role, msg, rubric),
                    reward=reward,
                    done=True,
                )

        # --- 6. Advance phase & switch speaker ---
        # Skip if plea handler already advanced the phase
        if not plea_handled:
            self._advance_phase(action, role)

        # --- 7. Check if judge should deliberate ---
        if s.phase == TrialPhase.JUDGE_DELIBERATION:
            self._judge_deliberate()
            # Apply terminal verdict alignment bonus
            self._apply_verdict_bonus()
            final_role = s.winner or role
            return VerdictStepResult(
                observation=self._build_observation(role, f"VERDICT: {s.verdict}", rubric),
                reward=self._last_reward,
                done=True,
                info={"verdict": s.verdict, "winner": s.winner.value if s.winner else "draw"},
            )

        msg = f"Your turn, {s.current_speaker.value.title()}. Phase: {s.phase.value}."
        return VerdictStepResult(
            observation=self._build_observation(s.current_speaker, msg, rubric),
            reward=reward,
            done=False,
        )

    @property
    def state(self) -> VerdictState:
        """Full episode state (server/judge eyes only)."""
        return self._state

    # ---- Internal helpers ------------------------------------------------

    def _build_observation(
        self, role: AgentRole, message: str = "",
        rubric: Optional[RubricScore] = None,
    ) -> VerdictObservation:
        """Build a PARTIALLY OBSERVABLE observation for the given role."""
        s = self._state
        private = s.prosecutor_evidence if role == AgentRole.PROSECUTOR else s.defense_evidence
        unrevealed = [e for e in private if not e.revealed]
        return VerdictObservation(
            case_id=s.case_id,
            case_brief=s.case_brief,
            role=role,
            phase=s.phase,
            turn_number=s.step_count,
            current_speaker=s.current_speaker,
            private_evidence=unrevealed,
            public_evidence=list(s.public_evidence),
            transcript=list(s.transcript),
            reward_breakdown=rubric,
            message=message,
        )

    def _handle_evidence_reveal(self, action: VerdictAction, role: AgentRole) -> None:
        """Move an evidence card from private to public."""
        s = self._state
        pool = s.prosecutor_evidence if role == AgentRole.PROSECUTOR else s.defense_evidence
        for card in pool:
            if card.evidence_id == action.evidence_id and not card.revealed:
                card.revealed = True
                s.public_evidence.append(card.model_copy())
                break

    def _handle_plea(self, action: VerdictAction, role: AgentRole) -> tuple[bool, str]:
        """Process plea bargain logic. Returns (episode_done, message)."""
        s = self._state
        is_plea = action.action_type == ActionType.PLEA
        if role == AgentRole.PROSECUTOR:
            s.prosecutor_plea = is_plea
        else:
            s.defense_plea = is_plea

        # Both sides have spoken on plea
        if s.prosecutor_plea is not None and s.defense_plea is not None:
            if s.prosecutor_plea and s.defense_plea:
                s.is_done = True
                s.phase = TrialPhase.EPISODE_DONE
                s.verdict = "Plea bargain accepted. Both parties settled."
                return True, "Plea accepted by both sides. Case settled."
            # At least one side rejected — proceed to trial
            s.phase = TrialPhase.OPENING_STATEMENTS
            s.current_speaker = AgentRole.PROSECUTOR
            return False, "Plea rejected. Proceeding to opening statements."

        # Only one side has spoken — toggle speaker for next plea turn
        if role == AgentRole.PROSECUTOR:
            s.current_speaker = AgentRole.DEFENSE
        else:
            s.current_speaker = AgentRole.PROSECUTOR
        return False, ""

    def _advance_phase(self, action: VerdictAction, role: AgentRole) -> None:
        """State machine: advance phase and toggle speaker."""
        s = self._state

        if s.phase == TrialPhase.PLEA_BARGAIN:
            # Toggle speaker within plea phase
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            return

        if s.phase == TrialPhase.OPENING_STATEMENTS:
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            else:
                s.phase = TrialPhase.ARGUMENT_ROUNDS
                s.current_speaker = AgentRole.PROSECUTOR
                s.round_number = 1
            return

        if s.phase == TrialPhase.ARGUMENT_ROUNDS:
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            else:
                s.round_number += 1
                if s.round_number > s.max_rounds:
                    s.phase = TrialPhase.CLOSING_STATEMENTS
                    s.current_speaker = AgentRole.PROSECUTOR
                else:
                    s.current_speaker = AgentRole.PROSECUTOR
            return

        if s.phase == TrialPhase.CLOSING_STATEMENTS:
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            else:
                s.phase = TrialPhase.JUDGE_DELIBERATION
            return

    def _judge_deliberate(self) -> None:
        """Compute final verdict based on accumulated rubric scores."""
        s = self._state
        p_total = sum(r.weighted_total for r in s.prosecutor_scores) if s.prosecutor_scores else 0.0
        d_total = sum(r.weighted_total for r in s.defense_scores) if s.defense_scores else 0.0
        p_avg = p_total / len(s.prosecutor_scores) if s.prosecutor_scores else 0.0
        d_avg = d_total / len(s.defense_scores) if s.defense_scores else 0.0

        if p_avg > d_avg + 0.05:
            s.winner = AgentRole.PROSECUTOR
            s.verdict = (
                f"The court finds in favor of the Prosecution. "
                f"(Avg score: Prosecution {p_avg:.3f} vs Defense {d_avg:.3f})"
            )
        elif d_avg > p_avg + 0.05:
            s.winner = AgentRole.DEFENSE
            s.verdict = (
                f"The court finds in favor of the Defense. "
                f"(Avg score: Defense {d_avg:.3f} vs Prosecution {p_avg:.3f})"
            )
        else:
            s.winner = None
            s.verdict = (
                f"The court declares a draw — neither side demonstrated clear superiority. "
                f"(Avg score: Prosecution {p_avg:.3f} vs Defense {d_avg:.3f})"
            )
        s.is_done = True
        s.phase = TrialPhase.EPISODE_DONE

    def _apply_verdict_bonus(self) -> None:
        """Apply terminal verdict_alignment bonus to the winning side's last score."""
        s = self._state
        if s.winner == AgentRole.PROSECUTOR and s.prosecutor_scores:
            s.prosecutor_scores[-1].verdict_alignment = 1.0
            self._last_reward += 0.15  # 15% weight terminal bonus
        elif s.winner == AgentRole.DEFENSE and s.defense_scores:
            s.defense_scores[-1].verdict_alignment = 1.0
            self._last_reward += 0.15
