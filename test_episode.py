"""
test_episode.py — Run one full courtroom episode end-to-end.
Simulates both Prosecutor and Defense taking turns through all phases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from models import ActionType, AgentRole, VerdictAction
from verdict_environment import VerdictEnvironment

def make_action(action_type: ActionType, thinking: str, argument: str, evidence_id=None) -> VerdictAction:
    return VerdictAction(
        thinking=thinking,
        action_type=action_type,
        argument=argument,
        evidence_id=evidence_id,
    )

def main():
    env = VerdictEnvironment(max_rounds=2)  # 2 rounds for quick test
    print("=" * 70)
    print("VERDICT — Full Episode Test")
    print("=" * 70)

    # --- RESET ---
    obs = env.reset()
    s = env.state
    print(f"\n📋 Case: {s.charge}")
    print(f"   Brief: {s.case_brief[:120]}...")
    print(f"   Phase: {obs.phase.value} | Speaker: {obs.current_speaker.value}")
    print(f"   Prosecutor evidence: {[e.title for e in s.prosecutor_evidence]}")
    print(f"   Defense evidence: {[e.title for e in s.defense_evidence]}")

    total_rewards = {"prosecutor": 0.0, "defense": 0.0}
    step_num = 0

    # --- PLEA PHASE: both reject ---
    print(f"\n{'─'*70}")
    print("PHASE: Plea Bargain")
    # Prosecutor rejects plea
    result = env.step(make_action(
        ActionType.ARGUE,
        "The evidence is strong. No reason to settle.",
        "The prosecution rejects any plea bargain. The evidence clearly shows wrongdoing and justice demands a full trial.",
    ))
    total_rewards["prosecutor"] += result.reward
    step_num += 1
    print(f"  Step {step_num} | Prosecutor rejects plea | reward={result.reward:.3f} | done={result.done}")

    # Defense rejects plea
    result = env.step(make_action(
        ActionType.ARGUE,
        "We are innocent. A trial will prove it.",
        "The defense rejects the plea. Our client is innocent and we will demonstrate this through evidence and testimony.",
    ))
    total_rewards["defense"] += result.reward
    step_num += 1
    print(f"  Step {step_num} | Defense rejects plea | reward={result.reward:.3f} | done={result.done}")
    print(f"  → Phase advanced to: {env.state.phase.value}")

    # --- OPENING STATEMENTS ---
    print(f"\n{'─'*70}")
    print("PHASE: Opening Statements")
    result = env.step(make_action(
        ActionType.ARGUE,
        "I need to establish the core narrative and reference the key facts.",
        "Ladies and gentlemen, the evidence will show a clear pattern of wrongdoing. "
        "The facts of this case are undeniable. Therefore, we will demonstrate through "
        "witness testimony and documentary evidence that the defendant acted with full knowledge.",
    ))
    total_rewards["prosecutor"] += result.reward
    step_num += 1
    print(f"  Step {step_num} | Prosecutor opening | reward={result.reward:.3f}")

    result = env.step(make_action(
        ActionType.ARGUE,
        "I must counter the prosecution's framing and establish reasonable doubt.",
        "The prosecution's narrative ignores critical context. However, the defense will show that "
        "every action taken was reasonable and justified. Furthermore, the evidence they rely on "
        "is circumstantial at best. We will present documentation proving our client's innocence.",
    ))
    total_rewards["defense"] += result.reward
    step_num += 1
    print(f"  Step {step_num} | Defense opening | reward={result.reward:.3f}")
    print(f"  → Phase advanced to: {env.state.phase.value}")

    # --- ARGUMENT ROUNDS ---
    print(f"\n{'─'*70}")
    print("PHASE: Argument Rounds")
    for rnd in range(1, env.state.max_rounds + 1):
        # Prosecutor argues
        p_ev = env.state.prosecutor_evidence
        ev_to_reveal = next((e for e in p_ev if not e.revealed), None)
        if ev_to_reveal and rnd == 1:
            result = env.step(make_action(
                ActionType.REVEAL_EVIDENCE,
                f"Revealing {ev_to_reveal.title} now to establish credibility early.",
                f"I present to the court: {ev_to_reveal.title}. {ev_to_reveal.description} "
                f"This evidence directly supports our case because it demonstrates a clear "
                f"pattern. Therefore, the defendant cannot deny these documented facts.",
                evidence_id=ev_to_reveal.evidence_id,
            ))
        else:
            result = env.step(make_action(
                ActionType.ARGUE,
                "The defense's counter was weak on the factual basis. I should press harder.",
                "The defense fails to address the core issue. On the contrary, their argument "
                "actually reinforces our position. The documented evidence shows a clear timeline "
                "of events. Consequently, no reasonable interpretation supports the defense's theory.",
            ))
        total_rewards["prosecutor"] += result.reward
        step_num += 1
        print(f"  Step {step_num} | Round {rnd} Prosecutor | reward={result.reward:.3f} | rubric={result.reward_breakdown.weighted_total:.3f}" if result.reward_breakdown else f"  Step {step_num} | Round {rnd} Prosecutor | reward={result.reward:.3f}")

        if result.done:
            break

        # Defense counters
        d_ev = env.state.defense_evidence
        d_reveal = next((e for e in d_ev if not e.revealed), None)
        if d_reveal and rnd == 2:
            result = env.step(make_action(
                ActionType.REVEAL_EVIDENCE,
                f"Time to play our card: {d_reveal.title}. This will undercut their argument.",
                f"The prosecution overlooks critical evidence. I present: {d_reveal.title}. "
                f"{d_reveal.description} This directly contradicts the prosecution's claims. "
                f"Nevertheless, even without this evidence, the prosecution has failed to meet "
                f"the burden of proof required.",
                evidence_id=d_reveal.evidence_id,
            ))
        else:
            result = env.step(make_action(
                ActionType.ARGUE,
                "I need to rebut the prosecutor's timeline argument and introduce doubt.",
                "The prosecution misrepresents the timeline. However, when we examine the actual "
                "sequence of events, a different picture emerges. Furthermore, the prosecution's "
                "own evidence contains inconsistencies that undermine their narrative. This fails "
                "to establish the clear causation they claim.",
            ))
        total_rewards["defense"] += result.reward
        step_num += 1
        print(f"  Step {step_num} | Round {rnd} Defense | reward={result.reward:.3f} | rubric={result.reward_breakdown.weighted_total:.3f}" if result.reward_breakdown else f"  Step {step_num} | Round {rnd} Defense | reward={result.reward:.3f}")

        if result.done:
            break

    # --- CLOSING STATEMENTS ---
    if not env.state.is_done:
        print(f"\n{'─'*70}")
        print("PHASE: Closing Statements")
        print(f"  Current phase: {env.state.phase.value}")

        result = env.step(make_action(
            ActionType.CLOSE,
            "Summarize the strongest points and end with impact.",
            "In closing, the prosecution has demonstrated through evidence and testimony that the "
            "defendant's actions were deliberate. The evidence presented — including documented records "
            "and expert analysis — leaves no reasonable doubt. We ask the court to find in our favor.",
        ))
        total_rewards["prosecutor"] += result.reward
        step_num += 1
        print(f"  Step {step_num} | Prosecutor closing | reward={result.reward:.3f}")

        if not result.done:
            result = env.step(make_action(
                ActionType.CLOSE,
                "End strong. Emphasize the holes in the prosecution's case.",
                "The defense has shown that the prosecution's case rests on circumstantial evidence "
                "and unfounded assumptions. However, the evidence we presented tells a different story. "
                "Therefore, we respectfully ask the court to rule in favor of the defense.",
            ))
            total_rewards["defense"] += result.reward
            step_num += 1
            print(f"  Step {step_num} | Defense closing | reward={result.reward:.3f}")

    # --- VERDICT ---
    print(f"\n{'='*70}")
    print("⚖️  VERDICT")
    print(f"{'='*70}")
    s = env.state
    print(f"  Done: {s.is_done}")
    print(f"  Phase: {s.phase.value}")
    print(f"  Verdict: {s.verdict}")
    print(f"  Winner: {s.winner.value if s.winner else 'DRAW'}")
    print(f"\n  Prosecutor total reward: {total_rewards['prosecutor']:.3f}")
    print(f"  Defense total reward:    {total_rewards['defense']:.3f}")
    print(f"  Steps taken:             {s.step_count}")
    print(f"  Transcript entries:      {len(s.transcript)}")

    # Rubric breakdown
    if s.prosecutor_scores:
        avg_p = sum(r.weighted_total for r in s.prosecutor_scores) / len(s.prosecutor_scores)
        print(f"\n  Prosecutor avg rubric:   {avg_p:.3f} ({len(s.prosecutor_scores)} scores)")
    if s.defense_scores:
        avg_d = sum(r.weighted_total for r in s.defense_scores) / len(s.defense_scores)
        print(f"  Defense avg rubric:      {avg_d:.3f} ({len(s.defense_scores)} scores)")

    print(f"\n✅ Episode completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
