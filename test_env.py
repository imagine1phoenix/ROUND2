from server.verdict_environment import VerdictEnvironment
import json

def test_environment():
    # 1. Setup mock case data
    case_data = {
        "charge": "Breach of Contract",
        "facts": "Company A claims Company B did not deliver software on time.",
        "evidence": [
            {"id": "ev_1", "description": "Signed contract with deadline of Jan 1.", "owner": "public"},
            {"id": "ev_2", "description": "Email from Company B asking for extension.", "owner": "prosecutor"}
        ]
    }
    
    env = VerdictEnvironment(case_data)
    obs = env.reset()
    print("Initial Obs:", json.dumps(obs, indent=2))
    
    # 2. Prosecutor turn
    prosecutor_xml = """
    <thinking>I should argue they missed the deadline and reveal my evidence.</thinking>
    <action>ARGUE</action>
    <argument>
    Company B clearly breached the contract. We have an email proving they asked for an extension after the fact.
    [Evidence: ev_2]
    </argument>
    """
    
    obs, reward, done, info = env.step(prosecutor_xml)
    print("\nProsecutor Reward:", json.dumps(info["detailed_rewards"], indent=2))
    
    # 3. Defense turn (Counters Prosecutor)
    # They should link to the previous statement ID (stmt_0)
    defense_xml = """
    <thinking>I need to counter their email argument.</thinking>
    <action>ARGUE</action>
    <argument>
    The extension was verbally agreed upon before the email.
    [Counter: stmt_0]
    [Evidence: ev_1]
    </argument>
    """
    
    obs, reward, done, info = env.step(defense_xml)
    print("\nDefense Reward:", json.dumps(info["detailed_rewards"], indent=2))
    
    print("\nFinal Transcript length:", len(env.transcript.statements))
    for s in env.transcript.statements:
        print("[{}] ID: {} | Action: {} | Linked: {} | Ev: {}".format(s.agent.upper(), s.id, s.action, s.linked_counter_id, s.evidence_used))

if __name__ == "__main__":
    test_environment()
