# LSTM Search Suggester — Implementation Plan

## Data Profile
- **850,001** query strings (song titles), avg **20.5 chars**, max **252**
- **39 unique chars**: space, `0–9`, `a–z` — already sanitized, no casing/punctuation issues
- Effective vocab with specials: **42 tokens** (`<PAD>`, `<BOS>`, `<EOS>` + 39 raw chars)

---

## File Structure

```
LSTMSearchSuggestor/
├── data/
│   └── chars16912451775209929649.txt   # raw training corpus (850k lines)
├── models/                             # saved checkpoints (.pt files)
├── src/
│   ├── vocab.py        # Vocab class: char→idx mapping, encode/decode, serialization
│   ├── dataset.py      # QueryDataset, collate_fn, train/val split
│   ├── model.py        # LSTMSuggestor nn.Module
│   ├── train.py        # training loop, MPS device setup, checkpointing
│   └── inference.py    # suggest() interface, beam search decoder
├── requirements.txt
└── CLAUDE.md
```

---

## Model Architecture

**Approach: character-level next-token prediction**
Each title is trained as: input = `<BOS> + title[:-1]`, target = `title[1:] + <EOS>`. The model learns to predict the next character at every position, which naturally enables auto-regressive completion at inference time.

| Component | Config | Rationale |
|---|---|---|
| Vocab size | 42 | 39 raw chars + PAD/BOS/EOS |
| Embedding dim | 64 | Small vocab needs small embeddings |
| LSTM hidden dim | 512 | Capacity for 850k varied sequences |
| LSTM layers | 2 | Standard depth; 3 risks over-smoothing on short seqs |
| Dropout | 0.3 | Applied between LSTM layers only (not after final layer) |
| Output | `Linear(512 → 42)` | Logits over vocab; loss ignores PAD index |

Forward pass shape flow:
```
[B, T] → Embedding → [B, T, 64] → LSTM → [B, T, 512] → Linear → [B, T, 42]
```
Hidden state `(h, c)` is carried across steps and exposed for incremental inference.

---

## Data Pipeline

**`vocab.py`**
- Build char→idx from corpus on first run, serialize to `models/vocab.json`
- Special token indices: `PAD=0`, `BOS=1`, `EOS=2`
- Methods: `encode(str) → List[int]`, `decode(List[int]) → str`

**`dataset.py`**
- `QueryDataset`: reads corpus, strips whitespace, filters empty lines
- Each item returns `(input_ids, target_ids)` as LongTensors
  - `input_ids  = [BOS] + encoded[:-1]`
  - `target_ids = encoded[1:]  + [EOS]`
- Sequences truncated at **100 chars** (covers ~99.9% of data; outlier 252-char titles are edge cases)
- `collate_fn`: pads batch to longest sequence in batch, returns lengths for `pack_padded_sequence`
- Train/val split: **90/10** random shuffle with fixed seed

**DataLoader config:**
- Batch size: **512** (small model + short sequences → fits MPS memory comfortably)
- `num_workers=0` (required on MPS; multiprocessing + MPS tensors causes deadlocks)
- `pin_memory=False` (MPS doesn't use pinned memory)

---

## Training Loop

**`train.py`**

```
Device setup:
  device = "mps" if torch.backends.mps.is_available() else "cpu"

Optimizer:   AdamW, lr=1e-3, weight_decay=1e-4
Scheduler:   CosineAnnealingLR(T_max=num_epochs)
Loss:        CrossEntropyLoss(ignore_index=PAD_IDX)
Grad clip:   max_norm=1.0 (essential for LSTM stability)
Epochs:      20 (early stopping on val loss, patience=3)
```

**Per-epoch loop:**
1. Forward pass with `pack_padded_sequence` / `pad_packed_sequence` (skip PAD tokens in LSTM)
2. Reshape logits to `[B*T, vocab]`, targets to `[B*T]` for loss
3. Backward + clip + step
4. Log train loss, val loss, val perplexity per epoch
5. Save checkpoint if val loss improves: `models/best.pt` (model state, vocab path, epoch, val_loss)

**Checkpoint format:**
```python
{
  "epoch": int,
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "val_loss": float,
  "vocab_path": "models/vocab.json"
}
```

---

## Inference Interface

**`inference.py` — `suggest(prefix, top_k=5, max_len=50) -> list[str]`**

1. **Encode prefix** → char indices, prepend `<BOS>`
2. **Warm up hidden state**: run prefix through LSTM step-by-step to get `(h, c)` at final prefix char — this "seeds" the model with query context
3. **Beam search** (width = `top_k`):
   - Maintain `top_k` beams as `(sequence, log_prob, hidden_state)`
   - At each step: expand each beam over all vocab, keep top `top_k` by cumulative log prob
   - Terminate beam when `<EOS>` is emitted or `max_len` reached
   - Normalize by sequence length to avoid length bias
4. **Decode** completed beams → strings, strip special tokens
5. Return list sorted by score (highest first)

**Key inference constraint:** hidden state `(h, c)` must be cloned per beam — beams diverge after the first generated character, so they cannot share state.

**CLI entry point:**
```
python src/inference.py --checkpoint models/best.pt --prefix "love " --top_k 5
```

---

## MPS-Specific Notes

- No `torch.cuda.amp.autocast` — use `torch.autocast(device_type="mps", dtype=torch.bfloat16)` if mixed precision is desired (PyTorch ≥ 2.2)
- `num_workers=0` on DataLoader — non-negotiable on MPS
- Move tensors to MPS before LSTM forward; avoid CPU↔MPS transfers inside the loop
- `pack_padded_sequence` requires sequences sorted by length descending and lengths on CPU (not MPS)
