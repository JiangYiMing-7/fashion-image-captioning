"""Tokenizer utilities for DeepFashion-MultiModal captions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Dict, Iterable, Optional

import torch

from data.text_utils import clean_caption

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - optional dependency
    spm = None

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None


class CaptionTokenizer:
    """Wrapper around SentencePiece or HuggingFace tokenizers."""

    def __init__(
        self,
        spm_model_path: Optional[str] = None,
        hf_name: Optional[str] = None,
        max_len: int = 28,
        lowercase: bool = True,
    ) -> None:
        if not spm_model_path and not hf_name:
            raise ValueError("Either spm_model_path or hf_name must be provided.")
        self.max_len = max_len
        self.lowercase = lowercase
        self.pad_token_id = 0
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None

        if spm_model_path:
            if spm is None:
                raise ImportError("sentencepiece is required for SentencePiece tokenizers.")
            self.backend = "spm"
            self.processor = spm.SentencePieceProcessor(model_file=spm_model_path)
            self.pad_token_id = max(self.processor.pad_id(), 0)
            self.bos_token_id = self.processor.bos_id() if self.processor.bos_id() >= 0 else None
            self.eos_token_id = self.processor.eos_id() if self.processor.eos_id() >= 0 else None
            self.tokenizer = None
        else:
            if AutoTokenizer is None:
                raise ImportError("transformers is required for HuggingFace tokenizers.")
            self.backend = "hf"
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
            self.processor = None
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            self.pad_token_id = int(self.tokenizer.pad_token_id)
            self.bos_token_id = int(self.tokenizer.bos_token_id) if self.tokenizer.bos_token_id is not None else None
            self.eos_token_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None

    @property
    def vocab_size(self) -> int:
        if self.backend == "spm":
            return int(self.processor.get_piece_size())
        return int(self.tokenizer.vocab_size)

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a caption and return tensors."""
        normalized = clean_caption(text, lowercase=self.lowercase)
        if not normalized:
            raise ValueError("Cannot encode an empty caption.")

        if self.backend == "spm":
            token_ids = self.processor.encode(normalized, out_type=int)
            token_ids = self._apply_special_tokens(token_ids)
            token_ids = token_ids[: self.max_len]
            input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
            attention_mask_tensor = torch.ones_like(input_ids_tensor, dtype=torch.long)
        else:
            encoded = self.tokenizer(
                normalized,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids_tensor = encoded["input_ids"].squeeze(0)
            attention_mask_tensor = encoded["attention_mask"].squeeze(0)

        input_ids_tensor = input_ids_tensor[: self.max_len]
        attention_mask_tensor = attention_mask_tensor[: input_ids_tensor.shape[0]]
        return {"input_ids": input_ids_tensor, "attention_mask": attention_mask_tensor}

    def _apply_special_tokens(self, token_ids: Iterable[int]) -> list[int]:
        tokens = list(token_ids)
        if self.bos_token_id is not None:
            tokens.insert(0, self.bos_token_id)
        if self.eos_token_id is not None:
            tokens.append(self.eos_token_id)
        return tokens


def _caption_iterator(train_jsonl: Path, lowercase: bool) -> Iterable[str]:
    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            caption = record.get("caption", "")
            cleaned = clean_caption(caption, lowercase=lowercase)
            if cleaned:
                yield cleaned


def train_sentencepiece_model(
    root: Path,
    output_dir: Path,
    vocab_size: int = 8000,
    character_coverage: float = 0.9995,
    model_type: str = "unigram",
    lowercase: bool = True,
) -> Path:
    """Train a SentencePiece model from processed captions."""
    if spm is None:
        raise ImportError("sentencepiece is required for training.")

    train_jsonl = root / "processed" / "train.jsonl"
    if not train_jsonl.is_file():
        raise FileNotFoundError(f"Processed train file not found: {train_jsonl}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "spm"

    # sentencepiece has trouble with non-ASCII paths on Windows.
    # Train in a temporary ASCII-safe directory then move the files back.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_prefix = Path(tmpdir) / "spm"
        spm.SentencePieceTrainer.train(
            sentence_iterator=_caption_iterator(train_jsonl, lowercase),
            model_prefix=str(tmp_prefix),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
        )
        for suffix in (".model", ".vocab"):
            src = tmp_prefix.with_suffix(suffix)
            dst = model_prefix.with_suffix(suffix)
            shutil.move(src, dst)
    return model_prefix.with_suffix(".model")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption tokenizer utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spm_parser = subparsers.add_parser("train_sentencepiece", help="Train a SentencePiece tokenizer.")
    spm_parser.add_argument("--root", type=str, default="./dataset")
    spm_parser.add_argument("--output-dir", type=str, default="./dataset/tokenizer")
    spm_parser.add_argument("--vocab-size", type=int, default=8000)
    spm_parser.add_argument("--character-coverage", type=float, default=0.9995)
    spm_parser.add_argument("--model-type", type=str, default="unigram")
    spm_parser.add_argument("--lowercase", action="store_true", dest="lowercase")
    spm_parser.add_argument("--no-lowercase", action="store_false", dest="lowercase")
    spm_parser.set_defaults(lowercase=True)
    return parser.parse_args()


def _main() -> None:
    args = _parse_cli_args()
    if args.command == "train_sentencepiece":
        model_path = train_sentencepiece_model(
            root=Path(args.root),
            output_dir=Path(args.output_dir),
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            model_type=args.model_type,
            lowercase=args.lowercase,
        )
        print(f"SentencePiece model saved to {model_path}")


if __name__ == "__main__":
    _main()


__all__ = ["CaptionTokenizer", "train_sentencepiece_model"]

