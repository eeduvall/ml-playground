import argparse

import torch
import torch.nn.functional as F

from vocab import Vocab, BOS_IDX, EOS_IDX
from model import LSTMSuggestor

Hidden = tuple[torch.Tensor, torch.Tensor]


def load_model(checkpoint_path: str, device: torch.device) -> tuple[LSTMSuggestor, Vocab]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = Vocab.from_file(ckpt["vocab_path"])
    model = LSTMSuggestor(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


def _clone_hidden(hidden: Hidden) -> Hidden:
    h, c = hidden
    return h.clone(), c.clone()


@torch.no_grad()
def suggest(
    prefix: str,
    model: LSTMSuggestor,
    vocab: Vocab,
    device: torch.device,
    top_k: int = 5,
    max_len: int = 50,
) -> list[str]:
    # Encode prefix and warm up hidden state across all prefix tokens
    ids = [BOS_IDX] + vocab.encode(prefix)
    hidden: Hidden | None = None
    for token_id in ids:
        x = torch.tensor([[token_id]], dtype=torch.long, device=device)
        logits, hidden = model.forward_step(x, hidden)

    # logits is the distribution over the first generated character
    log_probs = F.log_softmax(logits[0], dim=-1)
    top_log_probs, top_ids = log_probs.topk(top_k)

    # Each beam: (cumulative_log_prob, generated_ids, hidden_state)
    beams: list[tuple[float, list[int], Hidden]] = [
        (top_log_probs[i].item(), [top_ids[i].item()], _clone_hidden(hidden))
        for i in range(top_k)
    ]

    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_len - 1):
        if not beams:
            break

        next_beams: list[tuple[float, list[int], Hidden]] = []
        for log_prob, seq, h in beams:
            if seq[-1] == EOS_IDX:
                score = log_prob / len(seq)
                completed.append((score, seq[:-1]))
                continue

            x = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
            step_logits, h_new = model.forward_step(x, h)
            step_log_probs = F.log_softmax(step_logits[0], dim=-1)
            top_step_lp, top_step_ids = step_log_probs.topk(top_k)

            for i in range(top_k):
                next_beams.append((
                    log_prob + top_step_lp[i].item(),
                    seq + [top_step_ids[i].item()],
                    _clone_hidden(h_new),
                ))

        # Keep the top_k beams by length-normalized score
        next_beams.sort(key=lambda b: b[0] / len(b[1]), reverse=True)
        beams = next_beams[:top_k]

    # Flush remaining open beams
    for log_prob, seq, _ in beams:
        if seq and seq[-1] == EOS_IDX:
            seq = seq[:-1]
        score = log_prob / max(len(seq), 1)
        completed.append((score, seq))

    completed.sort(reverse=True)
    return [prefix + vocab.decode(seq) for _, seq in completed[:top_k]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab = load_model(args.checkpoint, device)

    results = suggest(args.prefix, model, vocab, device, args.top_k, args.max_len)
    print(f"\nSuggestions for '{args.prefix}':")
    for i, s in enumerate(results, 1):
        print(f"  {i}. {s}")
