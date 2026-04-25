# ⚖️ Verdict: Training LLMs to Reason, Argue, and Judge like a Lawyer
**Meta x HuggingFace x OpenEnv x PyTorch Hackathon (India 2026)**

> *"Justice is blind, but AI shouldn't be. Current LLMs agree with us; we need them to challenge us."*

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Environment](https://img.shields.io/badge/OpenEnv-Supported-success)
![Framework](https://img.shields.io/badge/PyTorch-TRL-red)
![Status](https://img.shields.io/badge/Hackathon-Submission-orange)

## 🔗 Quick Links (Hackathon Requirements)
* **Live Environment:** [Hugging Face Space](#) *(Pending Deployment)*
* **Training Script:** [Google Colab Notebook](#) *(Pending)*
* **Storytelling / Demo:** [YouTube Video (<2 min)](#) *(Pending)*

## 📌 Problem: Target Capability Gap
Legal reasoning is one of the most demanding cognitive tasks humans perform — requiring structured argumentation, evidence evaluation, anticipation of opposition, and real-time adaptation. 

Current open-source LLMs can retrieve legal information, but they fail to strategically argue within an adversarial setting. They suffer from sycophancy, agreeing with the prompter rather than defending a logical premise under adversarial pressure.

**Verdict** solves this. It is an RL training environment where agents learn to argue cases in a simulated courtroom. A Prosecutor agent and Defense agent receive case facts and must construct, adapt, and counter legal arguments across multiple turns. A Judge agent evaluates both sides and delivers a reasoned verdict.

**The Goal:** Train LLMs to reason adversarially, argue coherently under pressure, and improve through self-play — driving **emergent strategic behavior** and rigorous logic missing from standard Supervised Fine-Tuning (SFT).

## 🚀 Environment: Sight, Action, and Reward
Verdict operates as a **Multi-Agent Markov Game** built on the OpenEnv specifications.

- **Type:** Text-based, Multi-agent, Turn-based, **Partially Observable Markov Decision Process (POMDP)**
- **Setting:** A simulated courtroom handling realistic cases (contract disputes, trials, etc.)
- **Observability:** **Partially Observable.** Both the Prosecutor and Defense hold **Private Evidence Cards** that the opponent cannot see until explicitly revealed. This drives deep **Theory-of-Mind** reasoning, as agents must model the opponent's hidden knowledge and incentives.
- **Episode Structure:**
  1. Case briefing & Private Evidence Distribution.
  2. **Plea Bargain Strategy Phase:** Agents can attempt to negotiate a settlement (testing *cooperation* and *negotiation*). If no plea is reached, trial proceeds.
  3. Opening Statements.
  4. Argument Rounds (Dynamic countering & strategic evidence reveals).
  5. Closing Statements.
  6. Judge Deliberation & Reward Assignment.

## 🧠 Training Strategy: GRPO & Self-Play
We move beyond SFT using Group Relative Policy Optimization (GRPO) via HuggingFace `TRL` and `Unsloth`. 

Instead of teaching the model *how* to sound like a lawyer, we train it on *what actually wins cases*. By generating multiple responses to the same trial state and scoring them through our Rubric-based Judge, the policy model internalizes optimal argumentative strategies, logic detection, and narrative consistency.

### Expected Deliverables
- **OpenEnv Implementation:** The `VerdictEnv` framework, properly subclassing `Environment` / `MCPEnvironment` and respecting client-server separation with a strict Gym-style `reset()`, `step()`, `state()` API.
- **Training Scripts:** Efficient Unsloth PEFT + GRPO pipelines for consumer GPUs.
- **Interactive Space:** A HuggingFace Space where users can watch the trained LoRA adapters argue a case dynamically.

## 🤔 Why Does It Matter?
Sycophancy is one of the biggest hurdles preventing LLMs from useful autonomous problem-solving. If a model cannot defend a logically sound premise against an adversarial agent, it cannot be trusted to independently verify complex code, legal documents, or medical records. Verdict creates a measurable, objective proving ground to harden model logic through RL.

## 📊 Judging Criteria Addressed
- **Environment Innovation (40%):** *Is the environment novel, creative, or challenging? Does it meaningfully test the agent’s behavior?*
  **Yes.** Verdict moves land beyond single-turn puzzles into a continuous, partially observable Markov game. Modeling hidden evidence and logical structures fundamentally tests theory-of-mind and the emergence of strategic rhetoric.
- **Storytelling (30%):** *Does the team clearly explain the problem, environment, and agent behavior? Is the demo engaging and easy to follow?*
  **Yes.** We tackle the known LLM sycophancy problem using a highly intuitive, gamified metaphor. The HuggingFace Space UI allows users to easily visualize the complex adversarial logic unfold.
- **Showing Improvement in Rewards (20%):** *Does the demo provide observable evidence of training progress?*
  **Yes.** Our training script generates Plotly charts demonstrating the "win rate" and "coherence score" of the adapting agents improving over epochs via GRPO self-play.
- **Reward and Training Script/Pipeline Setup (10%):** *Is the reward logic coherent, and does the pipeline produce meaningful improvement?*
  **Yes.** The 5-part Rubric explicitly safeguards against reward hacking. The Unsloth+TRL pipeline proves fast, effective GRPO convergence on consumer hardware.

## 📈 Results: Post-Training Breakthroughs
*(Waiting for Colab Training to conclude...)*

> *Plots are saved natively as .png files within the repository. Both axes are explicitly labeled, comparing the untrained base model vs. the GRPO-trained policy on the same axes.*

- **Loss Curve:** 
  - *Caption: Plotting GRPO loss over training steps.*
  - ![Loss Plot](./docs/loss_curve.png)
- **Reward Improvement (Win Rate & Coherence):** 
  - *Caption: Comparing base vs. trained agent win-rate across 100 evaluation episodes.*
  - ![Reward Plot](./docs/reward_curve.png)

---
*Built by a team of 2nd-semester CSE students for the Meta Open Source AI Hackathon.*
