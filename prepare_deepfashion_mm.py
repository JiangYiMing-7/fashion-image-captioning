"""Prepare DeepFashion-MultiModal annotations for the unified pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets.text_utils import clean_caption, split_sentences

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default="./data/DeepFashion-MultiModal")
    parser.add_argument("--image-dir", type=str, default="image")
    parser.add_argument("--caption-file", type=str, default=None)
    parser.add_argument("--caption-dir", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lowercase", action="store_true", dest="lowercase", help="Force lowercase captions.")
    parser.add_argument(
        "--no-lowercase",
        action="store_false",
        dest="lowercase",
        help="Keep original casing during cleaning.",
    )
    parser.set_defaults(lowercase=True)
    return parser.parse_args()


def discover_annotation_files(
    root: Path, caption_file: Optional[str], caption_dir: Optional[str]
) -> List[Path]:
    files: List[Path] = []
    if caption_file:
        candidate = Path(caption_file)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Caption file not found: {candidate}")
        files.append(candidate)

    if caption_dir:
        cand_dir = Path(caption_dir)
        if not cand_dir.is_absolute():
            cand_dir = root / cand_dir
        if not cand_dir.is_dir():
            raise FileNotFoundError(f"Caption directory not found: {cand_dir}")
        for path in sorted(cand_dir.glob("*.json")):
            if path.is_file():
                files.append(path)

    if not files:
        raise ValueError("No caption files discovered. Provide --caption-file or --caption-dir.")
    return files


def build_image_lookup(image_root: Path, root: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    for path in image_root.rglob("*"):
        if path.is_file():
            lookup[path.name] = path.relative_to(root)
    return lookup


def _coerce_caption_list(value: Any) -> List[str]:
    captions: List[str] = []
    if isinstance(value, str):
        captions.append(value)
    elif isinstance(value, (int, float)):
        captions.append(str(value))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                captions.append(item)
            elif isinstance(item, (int, float)):
                captions.append(str(item))
    elif isinstance(value, dict):
        for key in ("caption", "text", "description", "sentence"):
            if key in value and isinstance(value[key], (str, list)):
                captions.extend(_coerce_caption_list(value[key]))
                break
    return captions


def _extract_image_path(item: Dict[str, Any]) -> Optional[str]:
    for key in ("image_path", "image", "img", "path", "file_name", "filename"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_image_id(item: Dict[str, Any], fallback: str) -> str:
    for key in ("image_id", "id", "uid", "guid"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return fallback


def extract_entries(
    data: Any, default_image_key: str = ""
) -> Iterable[Tuple[Optional[str], str, str]]:
    if isinstance(data, list):
        for record in data:
            if not isinstance(record, dict):
                continue
            captions = []
            for field in ("caption", "text", "description", "sentence", "sentences"):
                if field in record:
                    captions.extend(_coerce_caption_list(record[field]))
            if not captions:
                continue
            image_path = _extract_image_path(record) or default_image_key
            image_id = _extract_image_id(record, Path(image_path).name if image_path else default_image_key)
            for caption in captions:
                yield image_path, caption, image_id
        return

    if isinstance(data, dict):
        for key, value in data.items():
            captions = _coerce_caption_list(value)
            if captions:
                image_id = str(key)
                image_path = key if isinstance(key, str) else None
                for caption in captions:
                    yield image_path, caption, image_id
        return

    LOGGER.warning("Unsupported JSON structure encountered; skipping.")


def load_caption_records(files: Sequence[Path]) -> List[Tuple[Optional[str], str, str]]:
    records: List[Tuple[Optional[str], str, str]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as err:
                raise ValueError(f"Failed to parse {path}") from err
        start = len(records)
        for entry in extract_entries(data):
            records.append(entry)
        LOGGER.info("Loaded %d raw captions from %s", len(records) - start, path)
    return records


def resolve_image_path(
    relative_candidate: Optional[str],
    root: Path,
    image_root: Path,
    name_lookup: Dict[str, Path],
) -> Optional[Path]:
    candidates: List[Path] = []
    if relative_candidate:
        candidate_path = Path(relative_candidate)
        if candidate_path.is_absolute():
            try:
                candidate_path = candidate_path.relative_to(root)
            except ValueError:
                candidate_path = Path(os.path.relpath(candidate_path, root))
        candidates.append(candidate_path)
        candidates.append(image_root.relative_to(root) / candidate_path.name)

    base_name = Path(relative_candidate).name if relative_candidate else ""
    if base_name and base_name in name_lookup:
        candidates.append(name_lookup[base_name])

    for rel_path in candidates:
        full_path = root / rel_path
        if full_path.exists():
            return rel_path
    return None


def prepare_dataset(
    root: Path,
    image_dir: str,
    caption_file: Optional[str],
    caption_dir: Optional[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    lowercase: bool,
) -> None:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.")

    ann_files = discover_annotation_files(root, caption_file, caption_dir)
    image_root = (root / image_dir).resolve()
    name_lookup = build_image_lookup(image_root, root)
    records = load_caption_records(ann_files)

    samples: List[Dict[str, Any]] = []
    for relative_candidate, raw_caption, image_id in records:
        resolved_rel = resolve_image_path(relative_candidate, root, image_root, name_lookup)
        if not resolved_rel:
            LOGGER.warning("Skipping %s: image path unresolved.", relative_candidate or image_id)
            continue

        cleaned = clean_caption(raw_caption, lowercase=lowercase)
        sentences = split_sentences(cleaned)
        if sentences:
            cleaned = " ".join(sentences)
        if not cleaned:
            LOGGER.warning("Skipping %s: empty caption after cleaning.", resolved_rel)
            continue

        sample = {
            "image_path": resolved_rel.as_posix(),
            "image_id": str(image_id),
            "caption": cleaned,
        }
        samples.append(sample)

    total = len(samples)
    if total == 0:
        raise RuntimeError("No valid samples were found.")

    rng = random.Random(seed)
    rng.shuffle(samples)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    splits = {
        "train": samples[:train_count],
        "val": samples[train_count : train_count + val_count],
        "test": samples[train_count + val_count :],
    }

    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in splits.items():
        output_path = processed_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for sample in split_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        LOGGER.info("Wrote %d samples to %s", len(split_samples), output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    root = Path(args.root).resolve()
    prepare_dataset(
        root=root,
        image_dir=args.image_dir,
        caption_file=args.caption_file,
        caption_dir=args.caption_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        lowercase=args.lowercase,
    )
    LOGGER.info("Preparation complete.")


if __name__ == "__main__":
    main()

