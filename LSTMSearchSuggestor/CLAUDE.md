# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Context

This is the `LSTMSearchSuggestor` project within the **ml-playground** monorepo — a personal portfolio of ML experiments organized by algorithm type. Each project manages its own virtual environment and dependencies.

## Development Setup

All `src/` commands must be run from the `LSTMSearchSuggestor/` directory so relative paths to `data/` and `models/` resolve correctly.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commands

```bash
# Train (builds vocab on first run, saves best.pt to models/)
python src/train.py

# Train with overrides
python src/train.py --epochs 30 --batch_size 256 --lr 5e-4

# Run inference (requires a trained checkpoint)
python src/inference.py --prefix "love " --top_k 5
python src/inference.py --checkpoint models/best.pt --prefix "dark" --top_k 10 --max_len 60
```

## Architecture

Character-level LSTM trained on 850k song title queries (39-char vocab + 3 special tokens = 42 total).

**Data flow:**
- `src/vocab.py` — builds `char↔idx` mapping from corpus; serializes to `models/vocab.json`
- `src/dataset.py` — `QueryDataset` wraps each title as `(input=[BOS]+chars, target=chars+[EOS])`; `collate_fn` pads and sorts by length for `pack_padded_sequence`
- `src/model.py` — `LSTMSuggestor`: Embedding(42,64) → LSTM(64,512,layers=2,dropout=0.3) → Linear(512,42); exposes `forward()` for batched training and `forward_step()` for single-step inference
- `src/train.py` — AdamW + CosineAnnealingLR + early stopping; checkpoints saved to `models/best.pt`
- `src/inference.py` — warms up hidden state by running the full prefix through the LSTM, then beam-searches (width=top_k) with length-normalized scoring

**MPS constraints** (Apple Silicon):
- `num_workers=0` on all DataLoaders — multiprocessing deadlocks with MPS tensors
- `pack_padded_sequence` lengths must stay on CPU (not moved to MPS)
- No `torch.cuda.amp`; use `torch.autocast(device_type="mps", dtype=torch.bfloat16)` if mixed precision is needed (PyTorch ≥ 2.2)

## Tech Stack
- PyTorch 2.4 (MPS backend)
- No external tokenizer — pure character-level vocab built from corpus
- Model artifacts saved as `.pt` (torch.save), not `.pkl`
