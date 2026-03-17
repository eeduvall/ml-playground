import json
from pathlib import Path

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
_SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]


class Vocab:
    def __init__(self, chars: list[str]):
        self.idx2char = _SPECIAL_TOKENS + chars
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}

    @classmethod
    def from_corpus(cls, corpus_path: str | Path) -> "Vocab":
        chars: set[str] = set()
        with open(corpus_path) as f:
            for line in f:
                chars.update(line.rstrip("\n"))
        chars.discard("\n")
        return cls(sorted(chars))

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocab":
        with open(path) as f:
            data = json.load(f)
        return cls(data["chars"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"chars": self.idx2char[len(_SPECIAL_TOKENS):]}, f)

    def encode(self, text: str) -> list[int]:
        return [self.char2idx.get(c, PAD_IDX) for c in text]

    def decode(self, indices: list[int]) -> str:
        skip = {PAD_IDX, BOS_IDX, EOS_IDX}
        return "".join(self.idx2char[i] for i in indices if i not in skip)

    def __len__(self) -> int:
        return len(self.idx2char)
