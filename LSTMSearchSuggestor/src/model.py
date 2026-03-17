import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

Hidden = tuple[torch.Tensor, torch.Tensor]


class LSTMSuggestor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,           # [B, T]
        lengths: torch.Tensor,     # [B] on CPU, sorted descending
        hidden: Hidden | None = None,
    ) -> tuple[torch.Tensor, Hidden]:
        embedded = self.embedding(x)  # [B, T, E]
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        output, hidden = self.lstm(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)  # [B, T, H]
        return self.fc(output), hidden  # [B, T, V]

    def forward_step(
        self,
        x: torch.Tensor,           # [B, 1]
        hidden: Hidden | None = None,
    ) -> tuple[torch.Tensor, Hidden]:
        embedded = self.embedding(x)          # [B, 1, E]
        output, hidden = self.lstm(embedded, hidden)
        return self.fc(output.squeeze(1)), hidden  # [B, V]
