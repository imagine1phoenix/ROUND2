"""
Verdict Environment Models
===========================
Pydantic data models for the multi-agent courtroom POMDP.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional, List, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    PLEA = "plea"
    ARGUE = "argue"
    OBJECT = "object"
    REVEAL_EVIDENCE = "reveal_evidence"
    CONCEDE = "concede"
    CLOSE = "close"


class AgentRole(str, Enum):
    PROSECUTOR = "prosecutor"
    DEFENSE = "defense"


class TrialPhase(str, Enum):
    CASE_BRIEFING = "case_briefing"
    PLEA_BARGAIN = "plea_bargain"
    OPENING_STATEMENTS = "opening_statements"
    ARGUMENT_ROUNDS = "argument_rounds"
    CLOSING_STATEMENTS = "closing_statements"
    JUDGE_DELIBERATION = "judge_deliberation"
    EPISODE_DONE = "episode_done"


# ---------------------------------------------------------------------------
#  Evidence & Transcript
# ---------------------------------------------------------------------------

class EvidenceCard(BaseModel):
    """A single piece of evidence held by one side."""
    evidence_id: str
    title: str
    description: str
    owner: AgentRole
    revealed: bool = False


class TranscriptEntry(BaseModel):
    """One turn in the courtroom transcript."""
    role: AgentRole
    action_type: ActionType
    argument: str = Field(default="", description="The text spoken in court")
    evidence_revealed: Optional[str] = Field(
        default=None, description="Evidence ID if revealed this turn"
    )
    phase: TrialPhase


class RubricScore(BaseModel):
    """The 5-dimension composable rubric reward breakdown.

    Each component scores 0.0 to 1.0 before weighting.
    Weights: coherence=0.30, evidence=0.20, counter=0.20,
             consistency=0.15, verdict=0.15
    """
    coherence: float = Field(default=0.0, ge=0.0, le=1.0,
                             description="Argument logical consistency (30%)")
    evidence_usage: float = Field(default=0.0, ge=0.0, le=1.0,
                                  description="Strategic use of evidence (20%)")
    counter_quality: float = Field(default=0.0, ge=0.0, le=1.0,
                                   description="Rebuttal of opponent's points (20%)")
    consistency: float = Field(default=0.0, ge=0.0, le=1.0,
                               description="No self-contradiction across turns (15%)")
    verdict_alignment: float = Field(default=0.0, ge=0.0, le=1.0,
                                     description="Did the judge rule in agent's favor? (15%)")

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
    """Action submitted by a Prosecutor or Defense agent each turn."""
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
        ..., description="The discrete action to take this turn",
    )
    argument: str = Field(
        ..., max_length=1500,
        description="The text presented to the court. Must not exceed ~200 words.",
    )
    evidence_id: Optional[str] = Field(
        default=None,
        description="ID of the private evidence card to reveal (required if action_type is REVEAL_EVIDENCE)",
    )


class VerdictObservation(BaseModel):
    """What an agent sees after each step (partially observable)."""
    case_id: str = Field(..., description="Unique case identifier for this episode")
    case_brief: str = Field(..., description="The public case facts visible to all agents")
    role: AgentRole = Field(..., description="Which side this agent is playing")
    phase: TrialPhase = Field(..., description="Current phase of the trial")
    turn_number: int = Field(default=0, description="Current turn within the episode")
    current_speaker: AgentRole = Field(..., description="Whose turn it is to act")
    private_evidence: list[EvidenceCard] = Field(
        default_factory=list,
        description="Agent's OWN private evidence cards (opponent cannot see these)",
    )
    public_evidence: list[EvidenceCard] = Field(
        default_factory=list,
        description="All evidence that has been revealed by either side",
    )
    transcript: list[TranscriptEntry] = Field(
        default_factory=list, description="Full argument history visible to this agent",
    )
    reward_breakdown: Optional[RubricScore] = Field(
        default=None,
        description="Per-turn rubric scores (populated after Judge evaluation)",
    )
    message: str = Field(default="", description="System message")


class VerdictState(BaseModel):
    """Full episode state — visible to the Judge and the environment server."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = Field(default=0)
    case_id: str = Field(default="")
    case_brief: str = Field(default="")
    charge: str = Field(default="")
    phase: TrialPhase = Field(default=TrialPhase.CASE_BRIEFING)
    current_speaker: AgentRole = Field(default=AgentRole.PROSECUTOR)
    round_number: int = Field(default=0)
    max_rounds: int = Field(default=4)
    prosecutor_evidence: list[EvidenceCard] = Field(default_factory=list)
    defense_evidence: list[EvidenceCard] = Field(default_factory=list)
    public_evidence: list[EvidenceCard] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    prosecutor_scores: list[RubricScore] = Field(default_factory=list)
    defense_scores: list[RubricScore] = Field(default_factory=list)
    prosecutor_plea: Optional[bool] = Field(default=None)
    defense_plea: Optional[bool] = Field(default=None)
    is_done: bool = Field(default=False)
    verdict: Optional[str] = Field(default=None)
    winner: Optional[AgentRole] = Field(default=None)


# ---------------------------------------------------------------------------
#  Simple Models (backward compatibility with test_env.py / reward.py)
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    id: str
    description: str
    owner: Literal["prosecutor", "defense", "public"]
    is_revealed: bool = False


class Statement(BaseModel):
    id: str
    agent: Literal["prosecutor", "defense", "judge"]
    action: Literal["PLEA", "REVEAL_EVIDENCE", "ARGUE", "OBJECT", "CONCEDE", "CLOSE", "VERDICT"]
    text: str
    evidence_used: List[str] = Field(default_factory=list)
    linked_counter_id: Optional[str] = Field(None)
    raw_xml: str = Field(..., description="The raw XML generated by the agent")


class Case(BaseModel):
    case_id: str
    charge: str
    facts: str
    evidence_pool: List[Evidence]


class Transcript(BaseModel):
    case: Case
    statements: List[Statement] = Field(default_factory=list)
    winner: Optional[Literal["prosecutor", "defense", "tie"]] = None

    def get_statement(self, statement_id: str) -> Optional[Statement]:
        for s in self.statements:
            if s.id == statement_id:
                return s
        return None
