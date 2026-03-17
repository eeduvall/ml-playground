---
name: pytorch-lstm-mps
description: >
  PyTorch LSTM development skill for Apple M1 MPS backend.
  Use when: implementing LSTM layers, writing training loops,
  debugging MPS device errors, configuring DataLoaders for M1,
  or optimizing inference for Apple Silicon.
---

# PyTorch LSTM on Apple M1 (MPS)

## Device Setup
Always use this pattern — never hardcode "cuda" or "cpu":
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
```

## MPS Gotchas
- Use float32 only — float16 has limited MPS support and silently falls back to CPU
- Set pin_memory=False in all DataLoaders
- Set num_workers=0 — MPS + multiprocessing causes crashes on macOS
- If an op is unsupported on MPS, set PYTORCH_ENABLE_MPS_FALLBACK=1 in shell

## DataLoader Pattern
```python
DataLoader(dataset, batch_size=64, pin_memory=False, num_workers=0, shuffle=True)
```

## LSTM Module Pattern
```python
class LSTMSuggester(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden
```

## Training Loop Checklist
- Move both inputs AND targets to device before forward pass
- Call optimizer.zero_grad() before loss.backward()
- Clip gradients: nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
- Log loss every N steps, not every step (MPS overhead)

## Known Failures
- [none yet — add entries as you encounter them]