import re
from typing import Dict, Any
from .models import Transcript, Statement

class RewardModel:
    def __init__(self):
        # Weights according to RULES.md
        self.weights = {
            "argument_coherence": 0.30,
            "evidence_usage": 0.20,
            "counter_quality": 0.20,
            "consistency": 0.15,
            "verdict_alignment": 0.15
        }

    def compute_reward(self, transcript: Transcript, latest_statement: Statement) -> Dict[str, float]:
        """
        Computes the 5-dimensional composable reward for the latest statement.
        """
        rewards = {
            "argument_coherence": 0.0,
            "evidence_usage": 0.0,
            "counter_quality": 0.0,
            "consistency": 0.0,
            "verdict_alignment": 0.0,
            "formatting_penalty": 0.0
        }
        
        # 1. Format Penalties (Rule-Based Anti-Gaming)
        if "<thinking>" not in latest_statement.raw_xml or "<argument>" not in latest_statement.raw_xml:
            rewards["formatting_penalty"] -= 1.0
        
        word_count = len(latest_statement.text.split())
        if word_count > 300:
            rewards["formatting_penalty"] -= 1.0
            
        valid_actions = ["PLEA", "REVEAL_EVIDENCE", "ARGUE", "OBJECT", "CONCEDE", "CLOSE", "VERDICT"]
        if latest_statement.action not in valid_actions:
            rewards["formatting_penalty"] -= 0.5
            
        # 2. Argument Coherence (Heuristic: structure, length, readability)
        # For a full implementation, an LLM-as-a-judge could be used here.
        # As a proxy heuristic: an argument should have a reasonable length (e.g. > 20 words).
        if 20 <= word_count <= 300:
            rewards["argument_coherence"] = 1.0
        elif word_count < 20:
            rewards["argument_coherence"] = max(0.0, word_count / 20.0)
            
        # 3. Evidence Usage
        # Checks if evidence_used matches actual evidence IDs in the case and are revealed.
        if latest_statement.evidence_used:
            valid_evidence_count = 0
            for ev_id in latest_statement.evidence_used:
                # Find evidence in the case pool
                evidence = next((e for e in transcript.case.evidence_pool if e.id == ev_id), None)
                if evidence and (evidence.is_revealed or evidence.owner in ["public", latest_statement.agent]):
                    valid_evidence_count += 1
            
            if valid_evidence_count > 0:
                # Score based on proportion of valid evidence cited
                rewards["evidence_usage"] = valid_evidence_count / len(latest_statement.evidence_used)
            else:
                rewards["evidence_usage"] = 0.0 # Evidence dumping without context / fake evidence
        else:
            # If no evidence used, score is 0 unless it's a phase that doesn't need it.
            rewards["evidence_usage"] = 0.0

        # 4. Counter Quality
        # If it's linked to a counter, is it a valid opponent's statement?
        if latest_statement.linked_counter_id:
            opponent_statement = transcript.get_statement(latest_statement.linked_counter_id)
            if opponent_statement and opponent_statement.agent != latest_statement.agent:
                rewards["counter_quality"] = 1.0
            else:
                rewards["counter_quality"] = 0.2 # Invalid link
        else:
            # If they didn't link a counter, they missed an opportunity to directly address.
            # (In a real debate, not every statement is a counter, but if they are expected to, we penalize)
            rewards["counter_quality"] = 0.5

        # 5. Consistency
        # Check against previous statements by the same agent. 
        # A simple proxy: penalize repetition of the exact same points (word overlap).
        agent_previous_statements = [s for s in transcript.statements if s.agent == latest_statement.agent and s.id != latest_statement.id]
        repetition_penalty = 0.0
        latest_words = set(latest_statement.text.lower().split())
        
        for prev in agent_previous_statements:
            prev_words = set(prev.text.lower().split())
            if not latest_words or not prev_words: continue
            overlap = len(latest_words.intersection(prev_words)) / min(len(latest_words), len(prev_words))
            if overlap > 0.8: # high overlap = repetitive
                repetition_penalty = max(repetition_penalty, overlap)
        
        rewards["consistency"] = 1.0 - repetition_penalty
        
        # 6. Verdict Alignment (Terminal binary boost)
        # Evaluated at the end of the episode if a winner is decided.
        if transcript.winner:
            if transcript.winner == latest_statement.agent:
                rewards["verdict_alignment"] = 1.0
            else:
                rewards["verdict_alignment"] = 0.0

        # Calculate total weighted reward
        total_reward = (
            self.weights["argument_coherence"] * rewards["argument_coherence"] +
            self.weights["evidence_usage"] * rewards["evidence_usage"] +
            self.weights["counter_quality"] * rewards["counter_quality"] +
            self.weights["consistency"] * rewards["consistency"] +
            self.weights["verdict_alignment"] * rewards["verdict_alignment"] +
            rewards["formatting_penalty"]
        )
        
        rewards["total"] = total_reward
        return rewards
