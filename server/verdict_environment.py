"""
Verdict Environment — OpenEnv v0.2.3 Compliant
=================================================
Multi-agent adversarial courtroom POMDP.
Subclasses openenv.core.env_server.Environment.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from typing import Dict, Any, Optional, List

from openenv.core.env_server import Environment

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
#  Case Loader
# ---------------------------------------------------------------------------

def _extract_title(description: str) -> str:
    for sep in [" showing ", " confirming ", " citing ", " documenting ",
                " from ", " stating ", " linking ", " with "]:
        low = description.lower()
        if sep in low:
            return description[:low.index(sep)].strip()
    return description[:57] + "..." if len(description) > 60 else description


def load_cases(difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load cases from cases.json, optionally filtered by difficulty."""
    cases_path = os.path.join(os.path.dirname(__file__), "..", "cases.json")
    if not os.path.exists(cases_path):
        return []
    with open(cases_path) as f:
        raw_cases = json.load(f)
    result = []
    for raw in raw_cases:
        if difficulty and raw.get("difficulty", "medium") != difficulty:
            continue
        p_ev = [EvidenceCard(evidence_id=f"P{i+1}", title=_extract_title(d),
                description=d, owner=AgentRole.PROSECUTOR)
                for i, d in enumerate(raw.get("prosecutor_evidence", []))]
        d_ev = [EvidenceCard(evidence_id=f"D{i+1}", title=_extract_title(d),
                description=d, owner=AgentRole.DEFENSE)
                for i, d in enumerate(raw.get("defense_evidence", []))]
        result.append({
            "case_id": raw.get("id", str(uuid.uuid4())),
            "title": raw.get("title", "Unknown Case"),
            "category": raw.get("category", "General"),
            "charge": raw.get("charge", "Unknown Charge"),
            "case_brief": raw.get("facts", ""),
            "prosecutor_evidence": p_ev, "defense_evidence": d_ev,
            "difficulty": raw.get("difficulty", "medium"),
        })
    return result


# ---------------------------------------------------------------------------
#  Environment — subclasses openenv.core.env_server.Environment
# ---------------------------------------------------------------------------

class VerdictEnvironment(Environment[VerdictAction, VerdictObservation, VerdictState]):
    """Multi-agent adversarial courtroom POMDP.

    Subclasses OpenEnv's Environment base class with proper
    reset(), step(), and state property implementations.
    """

    def __init__(self, max_rounds: int = 4, difficulty: Optional[str] = None):
        super().__init__()
        self._max_rounds = max_rounds
        self._difficulty = difficulty
        self._state = VerdictState(max_rounds=max_rounds)
        self._cases = load_cases(difficulty)

    @property
    def state(self) -> VerdictState:
        """Get the current environment state (OpenEnv API)."""
        return self._state

    # -------------------------------------------------------------------
    #  reset() — OpenEnv API
    # -------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> VerdictObservation:
        """Reset the environment for a new episode (OpenEnv API)."""
        if seed is not None:
            random.seed(seed)

        case_data = kwargs.get("case_data")
        if case_data:
            case = case_data
        elif self._cases:
            case = random.choice(self._cases)
        else:
            case = {"case_id": "FALLBACK", "charge": "Unknown", "case_brief": "",
                    "prosecutor_evidence": [], "defense_evidence": []}

        self._state = VerdictState(
            episode_id=episode_id or str(uuid.uuid4()),
            max_rounds=self._max_rounds,
            case_id=case.get("case_id", ""),
            case_brief=case.get("case_brief", ""),
            charge=case.get("charge", ""),
            prosecutor_evidence=[e.model_copy() for e in case.get("prosecutor_evidence", [])],
            defense_evidence=[e.model_copy() for e in case.get("defense_evidence", [])],
            phase=TrialPhase.PLEA_BARGAIN,
            current_speaker=AgentRole.PROSECUTOR,
        )
        return self._get_observation(AgentRole.PROSECUTOR)

    # -------------------------------------------------------------------
    #  step() — OpenEnv API
    # -------------------------------------------------------------------

    def step(
        self,
        action: VerdictAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> VerdictObservation:
        """Process one agent action and advance the state (OpenEnv API).

        Returns VerdictObservation with `done`, `reward`, and `reward_breakdown`.
        """
        s = self._state
        role = s.current_speaker

        # Record transcript
        entry = TranscriptEntry(
            role=role, action_type=action.action_type, argument=action.argument,
            evidence_revealed=(action.evidence_id if action.action_type == ActionType.REVEAL_EVIDENCE else None),
            phase=s.phase,
        )
        s.transcript.append(entry)
        s.step_count += 1

        # Handle special actions
        if action.action_type == ActionType.REVEAL_EVIDENCE and action.evidence_id:
            self._reveal_evidence(role, action.evidence_id)
        if action.action_type == ActionType.PLEA:
            self._handle_plea(role)
        if action.action_type == ActionType.CONCEDE:
            s.is_done = True
            s.phase = TrialPhase.EPISODE_DONE
            s.winner = AgentRole.DEFENSE if role == AgentRole.PROSECUTOR else AgentRole.PROSECUTOR
            s.verdict = f"{role.value.title()} conceded the case."

        # Compute rubric
        rubric = self._compute_rubric(action, role)
        (s.prosecutor_scores if role == AgentRole.PROSECUTOR else s.defense_scores).append(rubric)
        reward = rubric.weighted_total

        # Advance phase
        if not s.is_done:
            self._advance_phase(role)
        if s.phase == TrialPhase.JUDGE_DELIBERATION:
            self._judge_deliberate()
            self._apply_verdict_bonus()
            reward = rubric.weighted_total  # recalculate after bonus

        # Build observation using OpenEnv fields (done, reward)
        obs = self._get_observation(role)
        obs.done = s.is_done
        obs.reward = reward
        obs.reward_breakdown = rubric
        return obs

    # -------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------

    def _get_observation(self, for_role: AgentRole) -> VerdictObservation:
        s = self._state
        priv = [e.model_copy() for e in
                (s.prosecutor_evidence if for_role == AgentRole.PROSECUTOR else s.defense_evidence)
                if not e.revealed]
        return VerdictObservation(
            case_id=s.case_id, case_brief=s.case_brief, role=for_role,
            phase=s.phase, turn_number=s.step_count, current_speaker=s.current_speaker,
            private_evidence=priv,
            public_evidence=[e.model_copy() for e in s.public_evidence],
            transcript=list(s.transcript),
            message=f"Your turn, {s.current_speaker.value}.",
        )

    def _advance_phase(self, role: AgentRole) -> None:
        s = self._state
        if s.phase == TrialPhase.PLEA_BARGAIN:
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            else:
                s.phase = TrialPhase.OPENING_STATEMENTS
                s.current_speaker = AgentRole.PROSECUTOR
        elif s.phase == TrialPhase.OPENING_STATEMENTS:
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            else:
                s.phase = TrialPhase.ARGUMENT_ROUNDS
                s.current_speaker = AgentRole.PROSECUTOR
                s.round_number = 1
        elif s.phase == TrialPhase.ARGUMENT_ROUNDS:
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            else:
                s.round_number += 1
                if s.round_number > s.max_rounds:
                    s.phase = TrialPhase.CLOSING_STATEMENTS
                s.current_speaker = AgentRole.PROSECUTOR
        elif s.phase == TrialPhase.CLOSING_STATEMENTS:
            if role == AgentRole.PROSECUTOR:
                s.current_speaker = AgentRole.DEFENSE
            else:
                s.phase = TrialPhase.JUDGE_DELIBERATION

    def _reveal_evidence(self, role, eid):
        s = self._state
        for ev in (s.prosecutor_evidence if role == AgentRole.PROSECUTOR else s.defense_evidence):
            if ev.evidence_id == eid and not ev.revealed:
                ev.revealed = True
                s.public_evidence.append(ev.model_copy())
                break

    def _handle_plea(self, role):
        s = self._state
        if role == AgentRole.PROSECUTOR: s.prosecutor_plea = True
        else: s.defense_plea = True
        if s.prosecutor_plea and s.defense_plea:
            s.is_done = True
            s.phase = TrialPhase.EPISODE_DONE
            s.verdict = "Both parties accepted a plea bargain."

    def _compute_rubric(self, action, role):
        s = self._state
        words = action.argument.split()
        wc = len(words)
        coherence = 1.0 if 20 <= wc <= 200 else max(0.0, wc/20.0) if wc < 20 else max(0.0, 1.0-(wc-200)/100)

        ev_score = 0.0
        if action.action_type == ActionType.REVEAL_EVIDENCE and action.evidence_id:
            ev_list = s.prosecutor_evidence if role == AgentRole.PROSECUTOR else s.defense_evidence
            ev_score = 1.0 if any(e.evidence_id == action.evidence_id for e in ev_list) else 0.0
        else:
            arg_l = action.argument.lower()
            refs = sum(1 for e in s.public_evidence if e.title.lower() in arg_l)
            ev_score = min(1.0, refs * 0.5) if refs else 0.0

        counter = 0.5
        opp = [t for t in s.transcript if t.role != role]
        if opp:
            ow = set(opp[-1].argument.lower().split())
            mw = set(action.argument.lower().split())
            counter = min(1.0, (len(mw & ow) / max(len(ow), 1)) * 3) if ow else 0.5

        consistency = 1.0
        my_prev = [t for t in s.transcript[:-1] if t.role == role]
        for p in my_prev:
            pw = set(p.argument.lower().split())
            mw = set(action.argument.lower().split())
            if mw and pw:
                ol = len(mw & pw) / min(len(mw), len(pw))
                if ol > 0.8: consistency = min(consistency, 1.0 - ol)

        return RubricScore(coherence=coherence, evidence_usage=ev_score,
                           counter_quality=counter, consistency=consistency)

    def _judge_deliberate(self):
        s = self._state
        p = sum(r.weighted_total for r in s.prosecutor_scores)/len(s.prosecutor_scores) if s.prosecutor_scores else 0
        d = sum(r.weighted_total for r in s.defense_scores)/len(s.defense_scores) if s.defense_scores else 0
        if p > d + 0.05:
            s.winner = AgentRole.PROSECUTOR
            s.verdict = f"Court finds for the Prosecution. (P:{p:.3f} vs D:{d:.3f})"
        elif d > p + 0.05:
            s.winner = AgentRole.DEFENSE
            s.verdict = f"Court finds for the Defense. (D:{d:.3f} vs P:{p:.3f})"
        else:
            s.verdict = f"Court declares a draw. (P:{p:.3f} vs D:{d:.3f})"
        s.is_done = True
        s.phase = TrialPhase.EPISODE_DONE

    def _apply_verdict_bonus(self):
        s = self._state
        if s.winner == AgentRole.PROSECUTOR and s.prosecutor_scores:
            s.prosecutor_scores[-1].verdict_alignment = 1.0
        elif s.winner == AgentRole.DEFENSE and s.defense_scores:
            s.defense_scores[-1].verdict_alignment = 1.0
