from server.models import Evidence, Statement, Case, Transcript
from server.reward import RewardModel
import json

def test_environment():
    # 1. Setup mock case data using the simple models
    case = Case(
        case_id="test-001",
        charge="Breach of Contract",
        facts="Company A claims Company B did not deliver software on time.",
        evidence_pool=[
            Evidence(id="ev_1", description="Signed contract with deadline of Jan 1.", owner="public"),
            Evidence(id="ev_2", description="Email from Company B asking for extension.", owner="prosecutor"),
        ]
    )
    transcript = Transcript(case=case)
    reward_model = RewardModel()

    print("Case:", case.charge)
    print("Evidence:", [e.id for e in case.evidence_pool])

    # 2. Prosecutor turn
    prosecutor_xml = """
    <thinking>I should argue they missed the deadline and reveal my evidence.</thinking>
    <action>ARGUE</action>
    <argument>
    Company B clearly breached the contract. We have an email proving they asked for an extension after the fact.
    [Evidence: ev_2]
    </argument>
    """

    import re

    def parse_xml(raw_xml, agent, turn_count):
        parsed = {"thinking": "", "action": "", "argument": "", "evidence_used": [], "linked_counter_id": None}
        m = re.search(r"<thinking>(.*?)</thinking>", raw_xml, re.DOTALL)
        if m: parsed["thinking"] = m.group(1).strip()
        m = re.search(r"<action>(.*?)</action>", raw_xml, re.IGNORECASE)
        if m: parsed["action"] = m.group(1).strip().upper()
        m = re.search(r"<argument>(.*?)</argument>", raw_xml, re.DOTALL)
        if m:
            text = m.group(1).strip()
            parsed["argument"] = text
            cm = re.search(r"\[Counter:\s*([\w-]+)\]", text)
            if cm: parsed["linked_counter_id"] = cm.group(1).strip()
            parsed["evidence_used"] = [e.strip() for e in re.findall(r"\[Evidence:\s*([\w-]+)\]", text)]
        stmt = Statement(
            id=f"stmt_{turn_count}", agent=agent, action=parsed.get("action", "ARGUE"),
            text=parsed.get("argument", ""), evidence_used=parsed.get("evidence_used", []),
            linked_counter_id=parsed.get("linked_counter_id"), raw_xml=raw_xml
        )
        return stmt

    stmt1 = parse_xml(prosecutor_xml, "prosecutor", 0)
    transcript.statements.append(stmt1)
    rewards1 = reward_model.compute_reward(transcript, stmt1)
    print("\nProsecutor Reward:", json.dumps(rewards1, indent=2))

    # 3. Defense turn
    defense_xml = """
    <thinking>I need to counter their email argument.</thinking>
    <action>ARGUE</action>
    <argument>
    The extension was verbally agreed upon before the email.
    [Counter: stmt_0]
    [Evidence: ev_1]
    </argument>
    """

    stmt2 = parse_xml(defense_xml, "defense", 1)
    transcript.statements.append(stmt2)
    rewards2 = reward_model.compute_reward(transcript, stmt2)
    print("\nDefense Reward:", json.dumps(rewards2, indent=2))

    print("\nFinal Transcript length:", len(transcript.statements))
    for s in transcript.statements:
        print("[{}] ID: {} | Action: {} | Linked: {} | Ev: {}".format(
            s.agent.upper(), s.id, s.action, s.linked_counter_id, s.evidence_used))

if __name__ == "__main__":
    test_environment()

