"""
Verdict Environment Models — OpenEnv v0.2.3
=============================================
Pydantic data models subclassing OpenEnv base types (Action, Observation, State).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional, List, Literal, Dict, Any

from pydantic import BaseModel, Field

# OpenEnv base classes
from openenv.core.env_server import (
    Action as OpenEnvAction,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)


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
    """The 5-dimension composable rubric reward breakdown."""
    coherence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_usage: float = Field(default=0.0, ge=0.0, le=1.0)
    counter_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    verdict_alignment: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def weighted_total(self) -> float:
        return (
            self.coherence * 0.30
            + self.evidence_usage * 0.20
            + self.counter_quality * 0.20
            + self.consistency * 0.15
            + self.verdict_alignment * 0.15
        )


# ---------------------------------------------------------------------------
#  OpenEnv Core Models — subclass Action, Observation, State
# ---------------------------------------------------------------------------

class VerdictAction(OpenEnvAction):
    """Action submitted by a Prosecutor or Defense agent each turn.

    Extends openenv.core.env_server.Action (adds `metadata: Dict`).
    """
    thinking: str = Field(
        ..., description="Internal monologue (Theory-of-Mind reasoning)",
    )
    action_type: ActionType = Field(
        ..., description="The discrete action to take this turn",
    )
    argument: str = Field(
        ..., max_length=1500,
        description="The text presented to the court (~200 words max)",
    )
    evidence_id: Optional[str] = Field(
        default=None,
        description="ID of evidence card to reveal (if action_type is REVEAL_EVIDENCE)",
    )


class VerdictObservation(OpenEnvObservation):
    """What an agent sees after each step (partially observable).

    Extends openenv.core.env_server.Observation (adds `done`, `reward`, `metadata`).
    """
    case_id: str = Field(default="", description="Unique case identifier")
    case_brief: str = Field(default="", description="Public case facts")
    role: AgentRole = Field(default=AgentRole.PROSECUTOR)
    phase: TrialPhase = Field(default=TrialPhase.CASE_BRIEFING)
    turn_number: int = Field(default=0)
    current_speaker: AgentRole = Field(default=AgentRole.PROSECUTOR)
    private_evidence: list[EvidenceCard] = Field(default_factory=list)
    public_evidence: list[EvidenceCard] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    reward_breakdown: Optional[RubricScore] = Field(default=None)
    message: str = Field(default="")


class VerdictState(OpenEnvState):
    """Full episode state — visible to the Judge and the environment server.

    Extends openenv.core.env_server.State (adds `episode_id`, `step_count`).
    """
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
