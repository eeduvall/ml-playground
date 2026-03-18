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


def _warmup(
    prefix: str,
    vocab: Vocab,
    model: LSTMSuggestor,
    device: torch.device,
) -> tuple[torch.Tensor, Hidden]:
    """Run prefix through the model; return (logits_for_first_gen_char, hidden)."""
    ids = [BOS_IDX] + vocab.encode(prefix)
    hidden: Hidden | None = None
    for token_id in ids:
        x = torch.tensor([[token_id]], dtype=torch.long, device=device)
        logits, hidden = model.forward_step(x, hidden)
    return logits[0], hidden


# ---------------------------------------------------------------------------
# Beam search (kept for reference / comparison)
# ---------------------------------------------------------------------------

def _diverse_topk(
    candidates: list[tuple[float, list[int], "Hidden"]],
    top_k: int,
    diversity_penalty: float,
) -> list[tuple[float, list[int], "Hidden"]]:
    """Greedily select top_k beams, penalizing repetition of the last token."""
    token_counts: dict[int, int] = {}
    scored: list[tuple[float, float, list[int], "Hidden"]] = []
    for lp, seq, h in candidates:
        last_tok = seq[-1]
        penalty = diversity_penalty * token_counts.get(last_tok, 0)
        adjusted = lp / len(seq) - penalty
        scored.append((adjusted, lp, seq, h))
        token_counts[last_tok] = token_counts.get(last_tok, 0) + 1
        if len(scored) == top_k * 3:
            break
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(lp, seq, h) for _, lp, seq, h in scored[:top_k]]


@torch.no_grad()
def suggest_beam(
    prefix: str,
    model: LSTMSuggestor,
    vocab: Vocab,
    device: torch.device,
    top_k: int = 5,
    max_len: int = 50,
    temperature: float = 1.0,
    diversity_penalty: float = 0.5,
) -> list[str]:
    logits, hidden = _warmup(prefix, vocab, model, device)

    log_probs = F.log_softmax(logits / temperature, dim=-1)
    n_candidates = min(top_k * 4, log_probs.size(-1))
    top_log_probs, top_ids = log_probs.topk(n_candidates)

    beams: list[tuple[float, list[int], Hidden]] = []
    seen: set[int] = set()
    for i in range(n_candidates):
        tok = top_ids[i].item()
        if tok not in seen:
            beams.append((top_log_probs[i].item(), [tok], _clone_hidden(hidden)))
            seen.add(tok)
        if len(beams) == top_k:
            break

    completed: list[tuple[float, list[int]]] = []

    for _ in range(max_len - 1):
        if not beams:
            break

        next_beams: list[tuple[float, list[int], Hidden]] = []
        for log_prob, seq, h in beams:
            if seq[-1] == EOS_IDX:
                completed.append((log_prob / len(seq), seq[:-1]))
                continue

            x = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
            step_logits, h_new = model.forward_step(x, h)
            step_log_probs = F.log_softmax(step_logits[0] / temperature, dim=-1)
            top_step_lp, top_step_ids = step_log_probs.topk(top_k)

            for i in range(top_k):
                next_beams.append((
                    log_prob + top_step_lp[i].item(),
                    seq + [top_step_ids[i].item()],
                    _clone_hidden(h_new),
                ))

        next_beams.sort(key=lambda b: b[0] / len(b[1]), reverse=True)
        beams = _diverse_topk(next_beams, top_k, diversity_penalty)

    for log_prob, seq, _ in beams:
        if seq and seq[-1] == EOS_IDX:
            seq = seq[:-1]
        completed.append((log_prob / max(len(seq), 1), seq))

    completed.sort(reverse=True)
    return [prefix + vocab.decode(seq) for _, seq in completed[:top_k]]


# ---------------------------------------------------------------------------
# Nucleus (top-p) sampling — each candidate generated independently
# ---------------------------------------------------------------------------

def _sample_one(
    init_logits: torch.Tensor,
    init_hidden: Hidden,
    model: LSTMSuggestor,
    device: torch.device,
    max_len: int,
    temperature: float,
    top_p: float,
) -> tuple[float, list[int]]:
    """Sample one completion via nucleus sampling; score with model log-probs."""
    seq: list[int] = []
    h = _clone_hidden(init_hidden)
    logits = init_logits
    model_log_prob = 0.0

    for _ in range(max_len):
        # Compute distribution with temperature
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        probs = log_probs.exp()

        # Nucleus filter: keep smallest set summing to >= top_p
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        nucleus_mask = (cumulative - sorted_probs) >= top_p
        filtered = sorted_probs.masked_fill(nucleus_mask, 0.0)
        filtered = filtered / filtered.sum()

        # Sample from filtered distribution
        sampled_rank = torch.multinomial(filtered, 1).item()
        tok = sorted_indices[sampled_rank].item()

        # Accumulate score using original (unfiltered) log-probs for fair ranking
        model_log_prob += log_probs[tok].item()
        seq.append(tok)

        if tok == EOS_IDX:
            seq = seq[:-1]
            break

        x = torch.tensor([[tok]], dtype=torch.long, device=device)
        step_logits, h = model.forward_step(x, h)
        logits = step_logits[0]

    score = model_log_prob / max(len(seq), 1)
    return score, seq


@torch.no_grad()
def suggest_nucleus(
    prefix: str,
    model: LSTMSuggestor,
    vocab: Vocab,
    device: torch.device,
    top_k: int = 5,
    max_len: int = 50,
    temperature: float = 1.2,
    top_p: float = 0.9,
) -> list[str]:
    logits, hidden = _warmup(prefix, vocab, model, device)

    # Generate top_k * 4 independent candidates, deduplicate, return top_k by score
    candidates: list[tuple[float, list[int]]] = []
    seen_seqs: set[str] = set()
    attempts = 0
    max_attempts = top_k * 20

    while len(candidates) < top_k * 4 and attempts < max_attempts:
        score, seq = _sample_one(logits, hidden, model, device, max_len, temperature, top_p)
        key = str(seq)
        if key not in seen_seqs:
            seen_seqs.add(key)
            candidates.append((score, seq))
        attempts += 1

    candidates.sort(reverse=True)
    return [prefix + vocab.decode(seq) for _, seq in candidates[:top_k]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--mode", choices=["beam", "nucleus"], default="nucleus",
                        help="Decoding strategy (default: nucleus)")
    # beam args
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Softmax temperature")
    parser.add_argument("--diversity_penalty", type=float, default=0.5,
                        help="[beam] Penalty per repeated last-token across sibling beams")
    # nucleus args
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="[nucleus] Nucleus probability mass cutoff")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, vocab = load_model(args.checkpoint, device)

    if args.mode == "nucleus":
        results = suggest_nucleus(args.prefix, model, vocab, device,
                                  args.top_k, args.max_len, args.temperature, args.top_p)
    else:
        results = suggest_beam(args.prefix, model, vocab, device,
                               args.top_k, args.max_len, args.temperature, args.diversity_penalty)

    print(f"\nSuggestions for '{args.prefix}' (mode={args.mode}):")
    for i, s in enumerate(results, 1):
        print(f"  {i}. {s}")
