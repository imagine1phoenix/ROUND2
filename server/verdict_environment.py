import re
import uuid
import json
from typing import Dict, Any, Tuple
from .models import Transcript, Statement, Case, Evidence
from .reward import RewardModel

class VerdictEnvironment:
    """
    Simulated Courtroom Environment handling state, action parsing, and reward distribution.
    Designed to be compatible with OpenEnv.
    """
    def __init__(self, case_data: Dict[str, Any]):
        self.reward_model = RewardModel()
        
        # Initialize Case
        evidence_pool = [Evidence(**e) for e in case_data.get("evidence", [])]
        self.case = Case(
            case_id=str(uuid.uuid4()),
            charge=case_data.get("charge", "Unknown Charge"),
            facts=case_data.get("facts", ""),
            evidence_pool=evidence_pool
        )
        self.transcript = Transcript(case=self.case)
        self.turn_count = 0
        self.max_turns = 10
        self.current_agent = "prosecutor"

    def reset(self) -> Dict[str, Any]:
        """Resets the environment for a new episode."""
        self.transcript.statements = []
        self.transcript.winner = None
        self.turn_count = 0
        self.current_agent = "prosecutor"
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        """Returns the Partially Observable Markov Decision Process (POMDP) state."""
        # Hide opponent's private evidence
        visible_evidence = [
            e.dict() for e in self.case.evidence_pool 
            if e.is_revealed or e.owner in ["public", self.current_agent]
        ]
        
        return {
            "agent": self.current_agent,
            "case_charge": self.case.charge,
            "public_facts": self.case.facts,
            "available_evidence": visible_evidence,
            "transcript_history": [s.dict() for s in self.transcript.statements],
            "turn": self.turn_count
        }

    def _parse_xml_action(self, raw_xml: str) -> Dict[str, Any]:
        """Parses the strict XML format required from agents."""
        parsed = {
            "thinking": "",
            "action": "",
            "argument": "",
            "evidence_used": [],
            "linked_counter_id": None
        }
        
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_xml, re.DOTALL)
        if thinking_match: parsed["thinking"] = thinking_match.group(1).strip()
            
        action_match = re.search(r"<action>(.*?)</action>", raw_xml, re.IGNORECASE)
        if action_match: parsed["action"] = action_match.group(1).strip().upper()
            
        argument_match = re.search(r"<argument>(.*?)</argument>", raw_xml, re.DOTALL)
        if argument_match: 
            argument_text = argument_match.group(1).strip()
            parsed["argument"] = argument_text
            
            # Simple heuristic to extract linked counter and evidence tags if they added them in text
            # For example: [Counter: stmt_123] or [Evidence: ev_456]
            counter_match = re.search(r"\[Counter:\s*([\w-]+)\]", argument_text)
            if counter_match: parsed["linked_counter_id"] = counter_match.group(1).strip()
                
            evidence_matches = re.findall(r"\[Evidence:\s*([\w-]+)\]", argument_text)
            parsed["evidence_used"] = [e.strip() for e in evidence_matches]
            
        return parsed

    def step(self, raw_xml: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Takes an agent's raw XML string, executes the turn, and computes reward."""
        parsed_action = self._parse_xml_action(raw_xml)
        
        statement_id = f"stmt_{self.turn_count}"
        
        statement = Statement(
            id=statement_id,
            agent=self.current_agent,
            action=parsed_action.get("action", "INVALID") if parsed_action.get("action") else "INVALID",
            text=parsed_action.get("argument", ""),
            evidence_used=parsed_action.get("evidence_used", []),
            linked_counter_id=parsed_action.get("linked_counter_id"),
            raw_xml=raw_xml
        )
        
        # Handle Reveal Evidence
        if statement.action == "REVEAL_EVIDENCE":
            for ev_id in statement.evidence_used:
                ev = next((e for e in self.case.evidence_pool if e.id == ev_id), None)
                if ev and ev.owner == self.current_agent:
                    ev.is_revealed = True
        
        self.transcript.statements.append(statement)
        
        # Compute Reward for the current step
        reward_dict = self.reward_model.compute_reward(self.transcript, statement)
        total_reward = reward_dict.get("total", 0.0)
        
        self.turn_count += 1
        done = self.turn_count >= self.max_turns or statement.action in ["CONCEDE", "VERDICT"]
        
        # Switch turns
        if not done:
            self.current_agent = "defense" if self.current_agent == "prosecutor" else "prosecutor"
            
        return self._get_observation(), total_reward, done, {"detailed_rewards": reward_dict}
