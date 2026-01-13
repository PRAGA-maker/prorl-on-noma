from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

@dataclass
class CharTokenizer:
    """A tiny character-level tokenizer for toy sequence tasks.

    Design goals:
    - deterministic
    - easy to inspect
    - stable vocab across runs
    """
    vocab: List[str]

    def __post_init__(self) -> None:
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: s for i, s in enumerate(self.vocab)}
        for tok in (PAD, BOS, EOS):
            if tok not in self.stoi:
                raise ValueError(f"Tokenizer vocab missing required token: {tok}")

    @property
    def pad_id(self) -> int: return self.stoi[PAD]
    @property
    def bos_id(self) -> int: return self.stoi[BOS]
    @property
    def eos_id(self) -> int: return self.stoi[EOS]
    @property
    def vocab_size(self) -> int: return len(self.vocab)

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in s:
            if ch not in self.stoi:
                raise ValueError(f"Unknown char {ch!r}. Add it to vocab.")
            ids.append(self.stoi[ch])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok in (PAD, BOS):
                continue
            if stop_at_eos and tok == EOS:
                break
            out.append(tok)
        return "".join(out)

def default_tokenizer() -> CharTokenizer:
    # Keep it intentionally small + stable.
    chars = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abcdefghijklmnopqrstuvwxyz") + list(":+-*/=()[],.?! ") + ["\n"]
    vocab = [PAD, BOS, EOS] + chars
    return CharTokenizer(vocab=vocab)
