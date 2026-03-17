import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

from vocab import Vocab, PAD_IDX, BOS_IDX, EOS_IDX

MAX_SEQ_LEN = 100


class QueryDataset(Dataset):
    def __init__(self, queries: list[str], vocab: Vocab):
        self.vocab = vocab
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.vocab.encode(self.queries[idx][:MAX_SEQ_LEN])
        input_ids = torch.tensor([BOS_IDX] + encoded, dtype=torch.long)
        target_ids = torch.tensor(encoded + [EOS_IDX], dtype=torch.long)
        return input_ids, target_ids


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch)

    lengths = torch.tensor([len(x) for x in inputs])
    lengths, sort_idx = lengths.sort(descending=True)

    inputs = pad_sequence(
        [inputs[i] for i in sort_idx], batch_first=True, padding_value=PAD_IDX
    )
    targets = pad_sequence(
        [targets[i] for i in sort_idx], batch_first=True, padding_value=PAD_IDX
    )
    return inputs, targets, lengths


def make_dataloaders(
    corpus_path: str | Path,
    vocab: Vocab,
    batch_size: int = 512,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    with open(corpus_path) as f:
        queries = [line.strip() for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(queries)

    n_val = int(len(queries) * val_fraction)
    train_ds = QueryDataset(queries[n_val:], vocab)
    val_ds = QueryDataset(queries[:n_val], vocab)

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,   # required on MPS
        pin_memory=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
