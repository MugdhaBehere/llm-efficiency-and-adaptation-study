# llm-efficiency-and-adaptation-study
This repository contains an empirical study on the efficiency and adaptation capacity of Low-Rank Adaptation (LoRA) for Large Language Models. Using TinyLlama-1.1B, I investigate how parameter-efficient fine-tuning performs in data-constrained regimes and analyze the trade-offs between adapter rank ($r$) and task-specific performance.

Research Objective
The goal of this project was to determine if a sub-2B parameter model could be effectively steered from a Causal Language Modeling (CLM) objective (continuation) to an Abstractive Summarization objective using minimal trainable parameters (~0.1%).

Tech Stack
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Library: Hugging Face peft, transformers, trl

Dataset: CNN/DailyMail (Subset)

Evaluation: ROUGE Metrics (1, 2, L)

Hardware: Trained on NVIDIA T4 GPU

Experimental Setup & Ablation :
I conducted a controlled experiment comparing a zero-shot base model against two LoRA configurations to observe the impact of the rank parameter ($r$):
Base Model: Zero-shot inference with a summarization prompt.
LoRA (r=8): Low-rank adaptation with 100 samples.
LoRA (r=32): Increased capacity adaptation to test for scaling bottlenecks.

Results and Quantitative Analysis:
Metric	Base Model	LoRA (r=8)	LoRA (r=32)
ROUGE-1	0.221	0.224	0.135
ROUGE-2	0.092	0.104	0.055
ROUGE-L	0.138	0.149	0.085


Key Findings:
The "Sweet Spot": The $r=8$ configuration yielded a 13% relative improvement in ROUGE-2, indicating successful learning of phrase-level summary structures.
Capacity Collapse: Increasing the rank to $r=32$ led to a significant performance drop. This suggests that in low-data regimes ($N=100$), higher-rank adapters overfit to the training noise, causing "catastrophic forgetting" of the base model's linguistic capabilities.
Continuation Bias: Qualitative analysis revealed a persistent "continuation bias," where the model occasionally treats the prompt as a prefix to be extended rather than a command to be executed.

How to Run:
1. Clone the repo:
git clone https://github.com/YourUsername/ProjectName.git

2. Install dependencies:
pip install transformers peft trl bitsandbytes evaluate rouge_score

3. Open LLM_Efficiency_and_adaptation_study.ipynb in Google Colab or Jupyter.

Future Work:
1. Data Scaling: Investigating the inflection point where $r=32$ begins to outperform $r=8$.
2. Instruction Tuning: Applying LoRA to models already pre-aligned with instructions (e.g., Gemma-2B-IT) to see if base alignment mitigates continuation bias.
3. Quantization: Implementing QLoRA to test even larger models (7B+) on consumer-grade hardware.
