---
name: lstm-lessons
description: >
  Experiment memory for the LSTM search suggester project.
  Use when: starting a new training run, debugging a recurring issue,
  choosing hyperparameters, or planning architecture changes.
  Contains lessons learned from past sessions — check this before
  making design decisions.
user-invocable: false
---

# LSTM Search Suggester — Experiment Lessons

## Architecture Decisions
- Character-level LSTM: Embedding(42,64) → LSTM(64,512,layers=2,dropout=0.3) → Linear(512,42)
- Beam search inference with length-normalized scoring

## Hyperparameter Findings
- [none yet]

## Known Failures
- **Beam search collapse (2026-03-18):** With prefix `"love "`, all top-5 suggestions were near-identical variants of "love is all i want for christmas is you [...]". The beam search is converging to a single high-probability path and its minor extensions, producing no diversity. Likely causes: (1) beam width too narrow relative to vocabulary branching, (2) length normalization not penalizing repetition, (3) model overtrained on a dominant pattern in corpus, or (4) no diversity penalty / temperature in sampling.

- **Diversity penalty + distinct initial tokens insufficient (2026-03-18):** Added temperature scaling (default 1.0), sibling last-token diversity penalty (default 0.5), and forced distinct first tokens across beams. Results improved slightly (shifted from "christmas" cluster to "love is on the way" cluster) but still highly duplicative — results 1–4 were all prefixes/extensions of the same path. **Root cause is likely in the model itself, not the decoding strategy**: the LSTM's hidden state after the prefix strongly concentrates probability on one continuation, so no inference-side penalty can fully overcome it. Beam search with any reasonable penalty will still collapse when the model's distribution is this peaked.

## What To Try Next
- ~~**Add temperature or top-p sampling**~~ — tried (default T=1.0, penalty=0.5); moved to a different dominant path but still collapsed
- ~~**Add a diversity penalty**~~ — tried sibling last-token penalty; marginally helpful but insufficient on its own
- **Replace beam search with stochastic sampling** — use top-p (nucleus) sampling with p=0.9 and temperature=1.1–1.3 to generate candidates independently, then rank by model score. This breaks the deterministic collapse entirely because each candidate is sampled independently rather than pruned from a shared tree.
- **Inspect corpus distribution** — check if "love is on the way" (and "love is all i want for christmas") variants dominate training data; if so the model is behaving correctly and the fix is data-side (deduplication or undersampling dominant patterns before retraining)
- **Retrain with label smoothing** (e.g. `smoothing=0.1` in `CrossEntropyLoss`) to prevent the model from concentrating so much probability on single continuations
- **Evaluate on multiple prefixes** to determine if collapse is prefix-specific ("love" → strong prior) or a general model problem
- **Try stochastic beam search**: at each step, sample (not argmax) from the top-k distribution weighted by temperature before expanding — breaks the deterministic path without fully abandoning beam structure
