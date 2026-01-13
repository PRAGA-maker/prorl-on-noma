from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import string

@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]
    pad_id: int
    bos_id: int
    eos_id: int

    @classmethod
    def build(cls) -> "CharTokenizer":
        # Fixed vocab that covers our toy prompts.
        special = ["<pad>", "<bos>", "<eos>"]
        chars = list("0123456789") + list("revpar:->") + ["e","v","n","o","d"," "]  # includes parity words + space
        # add any missing unique chars
        uniq = []
        for c in chars:
            if c not in uniq:
                uniq.append(c)
        vocab = special + uniq
        stoi = {c:i for i,c in enumerate(vocab)}
        itos = {i:c for c,i in stoi.items()}
        return cls(stoi=stoi, itos=itos, pad_id=stoi["<pad>"], bos_id=stoi["<bos>"], eos_id=stoi["<eos>"])

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in s:
            if ch not in self.stoi:
                raise ValueError(f"Unknown char {ch!r}. Add it to tokenizer vocab.")
            ids.append(self.stoi[ch])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos[int(i)])
        return "".join(out)
