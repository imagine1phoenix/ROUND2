"""
Verdict Environment — Phase-based Courtroom POMDP
====================================================
Multi-agent adversarial courtroom simulation with composable rubric rewards.
Compatible with OpenEnv v0.2.3.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

try:
    from .models import (
        ActionType, AgentRole, TrialPhase,
        EvidenceCard, TranscriptEntry, RubricScore,
        VerdictAction, VerdictObservation, VerdictState,
    )
except ImportError:
    from models import (
        ActionType, AgentRole, TrialPhase,
        EvidenceCard, TranscriptEntry, RubricScore,
        VerdictAction, VerdictObservation, VerdictState,
    )


# ---------------------------------------------------------------------------
#  Step Result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    observation: VerdictObservation
    reward: float
    done: bool


# ---------------------------------------------------------------------------
#  Case Loader
# ---------------------------------------------------------------------------

def _extract_title(description: str) -> str:
    """Extract a short title from an evidence description."""
    for sep in [" showing ", " confirming ", " citing ", " documenting ",
                " from ", " stating ", " linking ", " with "]:
        low = description.lower()
        if sep in low:
            idx = low.index(sep)
            return description[:idx].strip()
    if len(description) > 60:
        return description[:57] + "..."
    return description


def load_cases(difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load cases from cases.json, optionally filtered by difficulty.

    Returns a list of dicts with structured EvidenceCard objects.
    """
    cases_path = os.path.join(os.path.dirname(__file__), "..", "cases.json")
    if not os.path.exists(cases_path):
        return []

    with open(cases_path) as f:
        raw_cases = json.load(f)

    result = []
    for raw in raw_cases:
        if difficulty and raw.get("difficulty", "medium") != difficulty:
            continue

        p_evidence = []
        for i, desc in enumerate(raw.get("prosecutor_evidence", [])):
            p_evidence.append(EvidenceCard(
                evidence_id=f"P{i + 1}",
                title=_extract_title(desc),
                description=desc,
                owner=AgentRole.PROSECUTOR,
                revealed=False,
            ))

        d_evidence = []
        for i, desc in enumerate(raw.get("defense_evidence", [])):
            d_evidence.append(EvidenceCard(
                evidence_id=f"D{i + 1}",
                title=_extract_title(desc),
                description=desc,
                owner=AgentRole.DEFENSE,
                revealed=False,
            ))

        result.append({
            "case_id": raw.get("id", str(uuid.uuid4())),
            "title": raw.get("title", "Unknown Case"),
            "category": raw.get("category", "General"),
            "charge": raw.get("charge", "Unknown Charge"),
            "case_brief": raw.get("facts", ""),
            "prosecutor_evidence": p_evidence,
            "defense_evidence": d_evidence,
            "difficulty": raw.get("difficulty", "medium"),
        })

    return result


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------

class VerdictEnvironment:
    """Multi-agent adversarial courtroom POMDP.

    Usage:
        env = VerdictEnvironment(max_rounds=2)
        obs = env.reset()
        result = env.step(action)
    """

    def __init__(self, max_rounds: int = 4, difficulty: Optional[str] = None):
        self._max_rounds = max_rounds
        self._difficulty = difficulty
        self._state = VerdictState(max_rounds=max_rounds)
        self._cases = load_cases(difficulty)
        self._last_reward = 0.0

    @property
    def state(self) -> VerdictState:
        return self._state

    # -----------------------------------------------------------------------
    #  Reset
    # -----------------------------------------------------------------------

    def reset(self, case_data: Optional[Dict] = None) -> VerdictObservation:
        """Reset the environment for a new episode."""
        if case_data:
            case = case_data
        elif self._cases:
            case = random.choice(self._cases)
        else:
            case = {
                "case_id": "FALLBACK",
                "charge": "Unknown Charge",
                "case_brief": "No case data available.",
                "prosecutor_evidence": [],
                "defense_evidence": [],
            }

        self._state = VerdictState(
            max_rounds=self._max_rounds,
            case_id=case.get("case_id", ""),
            case_brief=case.get("case_brief", ""),
            charge=case.get("charge", ""),
            prosecutor_evidence=[e.model_copy() for e in case.get("prosecutor_evidence", [])],
            defense_evidence=[e.model_copy() for e in case.get("defense_evidence", [])],
            phase=TrialPhase.PLEA_BARGAIN,
            current_speaker=AgentRole.PROSECUTOR,
        )
        self._last_reward = 0.0
        return self._get_observation(AgentRole.PROSECUTOR)

    # -----------------------------------------------------------------------
    #  Step
    # -----------------------------------------------------------------------

    def step(self, action: VerdictAction) -> StepResult:
        """Process one agent action and advance the state."""
        s = self._state
        role = s.current_speaker

        # Record in transcript
        entry = TranscriptEntry(
            role=role,
            action_type=action.action_type,
            argument=action.argument,
            evidence_revealed=(
                action.evidence_id
                if action.action_type == ActionType.REVEAL_EVIDENCE
                else None
            ),
            phase=s.phase,
        )
        s.transcript.append(entry)
        s.step_count += 1

        # Handle evidence reveal
        if action.action_type == ActionType.REVEAL_EVIDENCE and action.evidence_id:
            self._reveal_evidence(role, action.evidence_id)

        # Handle plea
        if action.action_type == ActionType.PLEA:
            self._handle_plea(role)

        # Handle concede
        if action.action_type == ActionType.CONCEDE:
            s.is_done = True
            s.phase = TrialPhase.EPISODE_DONE
            s.winner = (
                AgentRole.DEFENSE if role == AgentRole.PROSECUTOR
                else AgentRole.PROSECUTOR
            )
            s.verdict = f"{role.value.title()} conceded the case."

        # Compute rubric score
        rubric = self._compute_rubric(action, role)
        if role == AgentRole.PROSECUTOR:
            s.prosecutor_scores.append(rubric)
        else:
            s.defense_scores.append(rubric)

        self._last_reward = rubric.weighted_total

        # Advance phase
        if not s.is_done:
            self._advance_phase(role)

        # Judge deliberation
        if s.phase == TrialPhase.JUDGE_DELIBERATION:
            self._judge_deliberate()
            self._apply_verdict_bonus()

        obs = self._get_observation(role)
        obs.reward_breakdown = rubric

        return StepResult(
            observation=obs,
            reward=self._last_reward,
            done=s.is_done,
        )

    # -----------------------------------------------------------------------
    #  Observation Builder
    # -----------------------------------------------------------------------

    def _get_observation(self, for_role: AgentRole) -> VerdictObservation:
        """Build a partially observable view for the given role."""
        s = self._state

        if for_role == AgentRole.PROSECUTOR:
            private_ev = [e.model_copy() for e in s.prosecutor_evidence if not e.revealed]
        else:
            private_ev = [e.model_copy() for e in s.defense_evidence if not e.revealed]

        public_ev = [e.model_copy() for e in s.public_evidence]

        return VerdictObservation(
            case_id=s.case_id,
            case_brief=s.case_brief,
            role=for_role,
            phase=s.phase,
            turn_number=s.step_count,
            current_speaker=s.current_speaker,
            private_evidence=private_ev,
            public_evidence=public_ev,
            transcript=list(s.transcript),
            message=f"Your turn, {s.current_speaker.value}.",
        )

    # -----------------------------------------------------------------------
    #  Phase Progression
    # -----------------------------------------------------------------------

    def _advance_phase(self, role: AgentRole) -> None:
        s = self._state

        if s.phase == TrialPhase.PLEA_BARGAIN:
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            else:
                s.phase = TrialPhase.OPENING_STATEMENTS
                s.current_speaker = AgentRole.PROSECUTOR
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

    # -----------------------------------------------------------------------
    #  Evidence
    # -----------------------------------------------------------------------

    def _reveal_evidence(self, role: AgentRole, evidence_id: str) -> None:
        s = self._state
        ev_list = (
            s.prosecutor_evidence if role == AgentRole.PROSECUTOR
            else s.defense_evidence
        )
        for ev in ev_list:
            if ev.evidence_id == evidence_id and not ev.revealed:
                ev.revealed = True
                s.public_evidence.append(ev.model_copy())
                break

    # -----------------------------------------------------------------------
    #  Plea Bargain
    # -----------------------------------------------------------------------

    def _handle_plea(self, role: AgentRole) -> None:
        s = self._state
        if role == AgentRole.PROSECUTOR:
            s.prosecutor_plea = True
        else:
            s.defense_plea = True

        if s.prosecutor_plea and s.defense_plea:
            s.is_done = True
            s.phase = TrialPhase.EPISODE_DONE
            s.verdict = "Both parties accepted a plea bargain."
            s.winner = None

    # -----------------------------------------------------------------------
    #  Rubric Scoring
    # -----------------------------------------------------------------------

    def _compute_rubric(self, action: VerdictAction, role: AgentRole) -> RubricScore:
        """Compute the 5-dimension rubric score for this turn."""
        s = self._state

        # 1. Coherence — argument structure and length
        words = action.argument.split()
        wc = len(words)
        if 20 <= wc <= 200:
            coherence = 1.0
        elif wc < 20:
            coherence = max(0.0, wc / 20.0)
        else:
            coherence = max(0.0, 1.0 - (wc - 200) / 100)

        # 2. Evidence usage
        evidence_score = 0.0
        ev_list = (
            s.prosecutor_evidence if role == AgentRole.PROSECUTOR
            else s.defense_evidence
        )
        if action.action_type == ActionType.REVEAL_EVIDENCE and action.evidence_id:
            valid = any(e.evidence_id == action.evidence_id for e in ev_list)
            evidence_score = 1.0 if valid else 0.0
        else:
            arg_lower = action.argument.lower()
            referenced = sum(
                1 for e in s.public_evidence if e.title.lower() in arg_lower
            )
            evidence_score = min(1.0, referenced * 0.5) if referenced else 0.0

        # 3. Counter quality — addressing opponent's last point
        counter_score = 0.5
        opponent_entries = [t for t in s.transcript if t.role != role]
        if opponent_entries:
            last_opp = opponent_entries[-1]
            opp_words = set(last_opp.argument.lower().split())
            my_words = set(action.argument.lower().split())
            if opp_words:
                overlap = len(my_words & opp_words) / len(opp_words)
                counter_score = min(1.0, overlap * 3)

        # 4. Consistency — penalize repetition
        my_entries = [t for t in s.transcript[:-1] if t.role == role]
        consistency = 1.0
        if my_entries:
            my_words = set(action.argument.lower().split())
            for prev in my_entries:
                prev_words = set(prev.argument.lower().split())
                if my_words and prev_words:
                    overlap = len(my_words & prev_words) / min(len(my_words), len(prev_words))
                    if overlap > 0.8:
                        consistency = min(consistency, 1.0 - overlap)

        # 5. Verdict alignment — set at terminal state only
        verdict_alignment = 0.0

        return RubricScore(
            coherence=coherence,
            evidence_usage=evidence_score,
            counter_quality=counter_score,
            consistency=consistency,
            verdict_alignment=verdict_alignment,
        )

    # -----------------------------------------------------------------------
    #  Judge Deliberation
    # -----------------------------------------------------------------------

    def _judge_deliberate(self) -> None:
        """Compute final verdict based on accumulated rubric scores."""
        s = self._state
        p_avg = (
            sum(r.weighted_total for r in s.prosecutor_scores) / len(s.prosecutor_scores)
            if s.prosecutor_scores else 0.0
        )
        d_avg = (
            sum(r.weighted_total for r in s.defense_scores) / len(s.defense_scores)
            if s.defense_scores else 0.0
        )

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
                f"The court declares a draw. "
                f"(Avg score: Prosecution {p_avg:.3f} vs Defense {d_avg:.3f})"
            )
        s.is_done = True
        s.phase = TrialPhase.EPISODE_DONE

    def _apply_verdict_bonus(self) -> None:
        """Apply terminal verdict_alignment bonus to the winning side."""
        s = self._state
        if s.winner == AgentRole.PROSECUTOR and s.prosecutor_scores:
            s.prosecutor_scores[-1].verdict_alignment = 1.0
            self._last_reward += 0.15
        elif s.winner == AgentRole.DEFENSE and s.defense_scores:
            s.defense_scores[-1].verdict_alignment = 1.0
            self._last_reward += 0.15
