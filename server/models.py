"""
Verdict Environment — Pydantic Models
======================================
Defines Action, Observation, and State dataclasses for the Verdict
courtroom RL environment, built on top of OpenEnv base types.

These models are shared between client and server.
Client code must ONLY import from this file — never from server internals.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Discrete action space for courtroom agents.

    NOTE: These are NOT OpenEnv reserved names (reset/step/state/close).
    """
    PLEA = "plea"
    ARGUE = "argue"
    OBJECT = "object"
    REVEAL_EVIDENCE = "reveal_evidence"
    CONCEDE = "concede"
    CLOSE = "close"


class AgentRole(str, Enum):
    """Which side of the courtroom the agent represents."""
    PROSECUTOR = "prosecutor"
    DEFENSE = "defense"


class TrialPhase(str, Enum):
    """Phase progression within a single episode."""
    CASE_BRIEFING = "case_briefing"
    PLEA_BARGAIN = "plea_bargain"
    OPENING_STATEMENTS = "opening_statements"
    ARGUMENT_ROUNDS = "argument_rounds"
    CLOSING_STATEMENTS = "closing_statements"
    JUDGE_DELIBERATION = "judge_deliberation"
    EPISODE_DONE = "episode_done"


# ---------------------------------------------------------------------------
#  Sub-models (building blocks)
# ---------------------------------------------------------------------------

class EvidenceCard(BaseModel):
    """A single piece of evidence, either public or private."""
    evidence_id: str = Field(..., description="Unique identifier for this evidence card")
    title: str = Field(..., description="Short label, e.g. 'CCTV Footage'")
    description: str = Field(..., description="Full text of the evidence")
    revealed: bool = Field(default=False, description="Whether this card is now public")


class TranscriptEntry(BaseModel):
    """One turn in the courtroom transcript."""
    turn_number: int
    role: AgentRole
    action_type: ActionType
    argument: str = Field(default="", description="The text spoken in court")
    evidence_revealed: Optional[str] = Field(default=None, description="Evidence ID if revealed this turn")
    phase: TrialPhase


class RubricScore(BaseModel):
    """The 5-dimension composable rubric reward breakdown.

    Each component scores 0.0 to 1.0 before weighting.
    Weights: coherence=0.30, evidence=0.20, counter=0.20,
             consistency=0.15, verdict=0.15
    """
    coherence: float = Field(default=0.0, ge=0.0, le=1.0, description="Argument logical consistency (30%)")
    evidence_usage: float = Field(default=0.0, ge=0.0, le=1.0, description="Strategic use of evidence (20%)")
    counter_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Rebuttal of opponent's points (20%)")
    consistency: float = Field(default=0.0, ge=0.0, le=1.0, description="No self-contradiction across turns (15%)")
    verdict_alignment: float = Field(default=0.0, ge=0.0, le=1.0, description="Did the judge rule in agent's favor? (15%)")

    @property
    def weighted_total(self) -> float:
        """Compute the final weighted reward scalar."""
        return (
            self.coherence * 0.30
            + self.evidence_usage * 0.20
            + self.counter_quality * 0.20
            + self.consistency * 0.15
            + self.verdict_alignment * 0.15
        )


# ---------------------------------------------------------------------------
#  OpenEnv Core Models
# ---------------------------------------------------------------------------

class VerdictAction(BaseModel):
    """Action submitted by a Prosecutor or Defense agent each turn.

    Agents MUST populate `thinking` and `argument` fields.
    The `action_type` selects the discrete move.
    `evidence_id` is required only when action_type == REVEAL_EVIDENCE.
    """
    thinking: str = Field(
        ...,
        description=(
            "Internal monologue (Theory-of-Mind): "
            "What is the opponent's weakest point? "
            "What hidden evidence might they hold? "
            "Do I reveal my evidence now?"
        ),
    )
    action_type: ActionType = Field(
        ...,
        description="The discrete action to take this turn",
    )
    argument: str = Field(
        ...,
        max_length=1500,  # ~200 words at ~7.5 chars/word
        description="The text presented to the court. Must not exceed ~200 words.",
    )
    evidence_id: Optional[str] = Field(
        default=None,
        description="ID of the private evidence card to reveal (required if action_type is REVEAL_EVIDENCE)",
    )


class VerdictObservation(BaseModel):
    """What an agent sees after each step.

    This is PARTIALLY OBSERVABLE: the agent only sees its own private
    evidence, never the opponent's hidden cards.
    """
    # --- Case context ---
    case_id: str = Field(..., description="Unique case identifier for this episode")
    case_brief: str = Field(..., description="The public case facts visible to all agents")
    role: AgentRole = Field(..., description="Which side this agent is playing")

    # --- Phase & turn tracking ---
    phase: TrialPhase = Field(..., description="Current phase of the trial")
    turn_number: int = Field(default=0, description="Current turn within the episode")
    current_speaker: AgentRole = Field(..., description="Whose turn it is to act")

    # --- Evidence (partial observability) ---
    private_evidence: list[EvidenceCard] = Field(
        default_factory=list,
        description="Agent's OWN private evidence cards (opponent cannot see these)",
    )
    public_evidence: list[EvidenceCard] = Field(
        default_factory=list,
        description="All evidence that has been revealed by either side",
    )

    # --- Transcript ---
    transcript: list[TranscriptEntry] = Field(
        default_factory=list,
        description="Full argument history visible to this agent",
    )

    # --- Reward signal (dense, per-turn) ---
    reward_breakdown: Optional[RubricScore] = Field(
        default=None,
        description="Per-turn rubric scores (populated after Judge evaluation)",
    )

    # --- System ---
    message: str = Field(
        default="",
        description="System message (e.g. 'Your turn, Prosecutor' or 'Plea accepted')",
    )


class VerdictState(BaseModel):
    """Full episode state — visible to the Judge and the environment server.

    This contains ALL information including both sides' private evidence.
    Agents never see VerdictState directly; they receive VerdictObservation.
    """
    # --- Episode metadata ---
    episode_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique episode identifier",
    )
    step_count: int = Field(default=0, description="Total steps taken this episode")

    # --- Case data ---
    case_id: str = Field(default="", description="Case identifier")
    case_brief: str = Field(default="", description="Public case facts")
    charge: str = Field(default="", description="The formal charge or dispute")

    # --- Phase machine ---
    phase: TrialPhase = Field(default=TrialPhase.CASE_BRIEFING)
    current_speaker: AgentRole = Field(default=AgentRole.PROSECUTOR)
    round_number: int = Field(default=0, description="Current argument round (1-4)")
    max_rounds: int = Field(default=4, description="Maximum argument rounds before closing")

    # --- Evidence (FULL — both sides, only visible to server/judge) ---
    prosecutor_evidence: list[EvidenceCard] = Field(default_factory=list)
    defense_evidence: list[EvidenceCard] = Field(default_factory=list)
    public_evidence: list[EvidenceCard] = Field(default_factory=list)

    # --- Transcript ---
    transcript: list[TranscriptEntry] = Field(default_factory=list)

    # --- Scoring ---
    prosecutor_scores: list[RubricScore] = Field(
        default_factory=list,
        description="Per-turn rubric scores for the Prosecutor",
    )
    defense_scores: list[RubricScore] = Field(
        default_factory=list,
        description="Per-turn rubric scores for the Defense",
    )

    # --- Plea bargain ---
    prosecutor_plea: Optional[bool] = Field(default=None, description="Did Prosecutor agree to plea?")
    defense_plea: Optional[bool] = Field(default=None, description="Did Defense agree to plea?")

    # --- Terminal ---
    is_done: bool = Field(default=False, description="Has the episode concluded?")
    verdict: Optional[str] = Field(default=None, description="Judge's final verdict text")
    winner: Optional[AgentRole] = Field(default=None, description="Which side won (None if plea or draw)")
