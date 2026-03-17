import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from vocab import Vocab, PAD_IDX
from dataset import make_dataloaders
from model import LSTMSuggestor


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for inputs, targets, lengths in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n_tokens = (targets != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for inputs, targets, lengths in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs, lengths)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        n_tokens = (targets != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / total_tokens


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True)
    vocab_path = models_dir / "vocab.json"

    if vocab_path.exists():
        vocab = Vocab.from_file(vocab_path)
        print(f"Loaded vocab ({len(vocab)} tokens)")
    else:
        vocab = Vocab.from_corpus(args.corpus)
        vocab.save(vocab_path)
        print(f"Built vocab ({len(vocab)} tokens) → {vocab_path}")

    train_loader, val_loader = make_dataloaders(
        args.corpus, vocab, batch_size=args.batch_size
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model_config = dict(
        vocab_size=len(vocab),
        embed_dim=64,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        pad_idx=PAD_IDX,
    )
    model = LSTMSuggestor(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = models_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ppl={math.exp(val_loss):.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "vocab_path": str(vocab_path),
                    "model_config": model_config,
                },
                checkpoint_path,
            )
            print(f"         Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping (patience={args.patience})")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/chars16912451775209929649.txt")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    main(parser.parse_args())
