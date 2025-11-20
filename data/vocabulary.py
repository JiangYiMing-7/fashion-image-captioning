from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]


@dataclass
class Vocabulary:
    min_freq: int = 1
    tokenizer: str = "simple"
    stoi: Dict[str, int] = field(default_factory=dict)
    itos: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.stoi:
            for token in SPECIAL_TOKENS:
                self._add_token(token)

    @property
    def pad_idx(self) -> int:
        return self.stoi["<pad>"]

    @property
    def start_idx(self) -> int:
        return self.stoi["<start>"]

    @property
    def end_idx(self) -> int:
        return self.stoi["<end>"]

    @property
    def unk_idx(self) -> int:
        return self.stoi["<unk>"]

    def __len__(self) -> int:
        return len(self.itos)

    def _add_token(self, token: str) -> None:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def tokenize(self, sentence: str) -> List[str]:
        if self.tokenizer == "jieba":
            import jieba

            return list(jieba.cut(sentence))
        return sentence.lower().strip().split()

    def build(self, captions: Iterable[str]) -> None:
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(self.tokenize(caption))

        for token, freq in counter.items():
            if freq >= self.min_freq:
                self._add_token(token)

    def numericalize(self, caption: str, max_length: int) -> List[int]:
        tokens = self.tokenize(caption)
        tokens = tokens[: max_length - 2]
        ids = [self.start_idx]
        ids += [self.stoi.get(token, self.unk_idx) for token in tokens]
        ids.append(self.end_idx)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        words = []
        for idx in ids:
            word = self.itos[idx]
            if word in {"<start>", "<end>", "<pad>"}:
                continue
            words.append(word)
        return " ".join(words).strip()

    def save(self, path: Path) -> None:
        payload = {"stoi": self.stoi, "itos": self.itos, "min_freq": self.min_freq, "tokenizer": self.tokenizer}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(min_freq=data["min_freq"], tokenizer=data["tokenizer"], stoi=data["stoi"], itos=data["itos"])

