# 📜 Verdict: Environment Mechanics & Rules

This document outlines the strict physics, action spaces, and reward mechanics of the `Verdict` OpenEnv environment. These rules ensure the RL environment is stable, computable, and bounded enough for rapid convergence during training.

## 1. Agent Design & Constraints
The environment runs on single base LLM utilizing separate LoRA adapters or system prompts for three distinct agent types.

### The Prosecutor
- **Objective:** Maximize the perceived guilt/liability of the subject.
- **Input:** Public Case facts, **Private Prosecutor Evidence**, Complete Trial Transcript up to current turn.

### The Defense
- **Objective:** Minimize the perceived guilt/liability of the subject by reframing the narrative and discrediting the Prosecutor.
- **Input:** Public Case facts, **Private Defense Evidence**, Complete Trial Transcript up to current turn.

### The Judge (Reward Model)
- **Objective:** Compute dense reward signals based on a multi-dimensional rubric.
- **Role:** Frozen LLM evaluator (not trained during the RL loop).

## 2. Action Space (XML Structured Formatting)
To bridge the gap between continuous text generation and discrete agent decision-making, agents **must** output their responses using strict XML tags. Formatting violations result in immediate negative rewards.

**Valid Output Format:**
```xml
<thinking>
[Internal Monologue: What is the opponent's weakest point? What hidden evidence might they hold (Theory-of-Mind)? Do I reveal my evidence now?]
</thinking>
<action>[PLEA | REVEAL_EVIDENCE | ARGUE | OBJECT | CONCEDE | CLOSE]</action>
<argument>
[The actual text presented to the court. Must not exceed 200 words.]
</argument>
```

## 3. The 5-Dimension Composable Rubric (OpenEnv)
To prevent reward hacking (e.g., repeating the same word to sound confident), the Judge calculates the reward based on 5 weighted pillars using OpenEnv's **Composable Rubric System**. This provides a rich, informative gradient rather than a sparse binary 0/1 signal.

| Component | Weight | Description | Scoring Logic (0.0 to 1.0)* |
| :--- | :--- | :--- | :--- |
| **Argument Coherence** | 30% | Internal consistency and logical flow. | Penalized for hallucinating fake laws. |
| **Evidence Usage** | 20% | Strategic and correct use of provided facts. | 0 if evidence dumping without context. |
| **Counter Quality** | 20% | Did the agent accurately address the previous turn? | Low score for 'talking past' the opponent. |
| **Consistency** | 15% | Did the agent contradict its own earlier claims? | Repetitive loops incur heavy penalties. |
| **Verdict Alignment** | 15% | Did the final Judge ruling favor this agent? | Binary boost (0 or 1) given at terminal state. |

*\*Note: For GRPO, these base heuristic scores are aggregated to formulate the advantage metric.*

### Rule-Based Format Penalties (Anti-Gaming)
- **-1.0 Reward:** Missing `<thinking>` or `<argument>` tags.
- **-1.0 Reward:** Outputting more than 300 words generated in the `<argument>` block.
- **-0.5 Reward:** Attempting an invalid discrete `<action>`.

## 4. Phase Progression & Core Mechanics

### 4.1. The Negotiation Mechanics (Plea Bargain)
Before entering arguments, agents can select `<action>PLEA</action>`. If both agents agree on plea terms, both receive a medium positive reward, avoiding the high-risk, zero-sum trial. This trains **cooperation and negotiation** behaviors in a predominantly competitive framework.

### 4.2. Rollout Logic for RL
For hackathon training efficiency, full 10-turn rollouts during RL are truncated. We train on **Partial Trial Transcripts**.
1. **State Sampling:** The environment initializes by randomly selecting a phase from synthetic trial histories (e.g., *Round 2, Prosecutor's Turn, with 1 Private Evidence card remaining*).
2. **Step:** The policy agent generates one move, demonstrating *theory-of-mind* regarding the opponent's potential hidden evidence.
3. **Reward:** The Judge/Rule-Engine computes the immediate multidimensional reward for that specific step.
4. **Update:** Backpropagation via GRPO occurs based on the group evaluation of that single step.

## 5. Curriculum Escalation Protocol
If an agent achieves >70% win rate (Judge scoring) on the current difficulty tier, the environment introduces harder synthetic cases:
- **Tier 1:** Single-charge cases, straightforward evidence.
- **Tier 2:** Multi-charge cases, conflicting statements.
- **Tier 3 (Asymmetric):** The agent is assigned a significantly weaker case fact-pattern and must rely entirely on rhetorical logic to minimize losses.

## 6. Hackathon Submission Checklist
To guarantee compliance with the Meta x HuggingFace x OpenEnv standards before the 5 PM deadline:
- [ ] **OpenEnv Integration:** Subclass and utilize the latest `OpenEnv` release.
- [ ] **Training Script:** Provide a minimal `.ipynb` training script via Unsloth or HF TRL (GRPO).
- [ ] **HF Spaces:** Deploy the OpenEnv compliant environment natively on Hugging Face Spaces.
- [ ] **Storytelling:** Produce a mini-blog (HF) or <2 min video (YouTube) presenting the submission.
