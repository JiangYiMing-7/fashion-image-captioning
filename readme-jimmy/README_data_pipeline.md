# DeepFashion-MultiModal Data Pipeline

Unified, model-agnostic data handling for the image captioning assignment. Every model (baseline RNN, local-attention hybrid, ViT-transformer) consumes the same processed annotations, tokenizer, dataset, collate function, and dataloaders.

## Directory Layout

```
data/
  prepare_deepfashion_mm.py      # preprocessing entry point
datasets/
  text_utils.py                  # caption cleaning helpers
  tokenizer.py                   # CaptionTokenizer + SentencePiece trainer
  deepfashion_mm_dataset.py      # torch.utils.data.Dataset implementation
  collate.py                     # unified batch collation
  datamodule.py                  # DeepFashionMMDataModule factory
scripts/
  debug_deepfashion_dataloader.py  # smoke test for loaders
README_data_pipeline.md
```

## 1. Build Processed Annotations

```
python data/prepare_deepfashion_mm.py ^
  --root ./data/DeepFashion-MultiModal ^
  --image-dir image ^
  --caption-dir textual_descriptions ^
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 ^
  --seed 0
```

Key behavior:

- Reads one or more JSON caption sources (`--caption-file` or `--caption-dir`).
- Cleans text via `datasets/text_utils.py`, joins multi-sentence captions, skips missing data.
- Splits deterministically and writes `processed/train|val|test.jsonl`.

## 2. Train the SentencePiece Tokenizer

```
python -m datasets.tokenizer train_sentencepiece ^
  --root ./data/DeepFashion-MultiModal ^
  --output-dir ./data/DeepFashion-MultiModal/tokenizer ^
  --vocab-size 8000
```

- Reads captions from `processed/train.jsonl`.
- Saves `spm.model` and `spm.vocab` into the output directory.
- You can swap in a HuggingFace tokenizer later by name (e.g., `bert-base-uncased`).

## 3. Use the Dataset & DataModule

```python
from datasets.datamodule import DeepFashionMMDataModule
from datasets.tokenizer import CaptionTokenizer

tokenizer = CaptionTokenizer(spm_model_path="./data/DeepFashion-MultiModal/tokenizer/spm.model")
dm = DeepFashionMMDataModule(
    root="./data/DeepFashion-MultiModal",
    tokenizer=tokenizer,
    batch_size=32,
    num_workers=8,
    image_size=384,
    max_len=28,
)

batch = next(iter(dm.train_dataloader()))
image = batch["image"]          # [B, 3, 384, 384]
input_ids = batch["input_ids"]  # [B, L] or None
captions = batch["caption"]     # list[str]
```

Behind the scenes:

- `DeepFashionMMDataset` loads normalized RGB images per sample, tokenizes captions if a tokenizer is supplied, and returns one unified dict.
- `deepfashion_collate_fn` stacks images, pads token sequences, and keeps metadata lists for compatibility across models.
- `DeepFashionMMDataModule` wires train/val/test dataloaders with consistent transforms (augmentation for train only), shuffling, and worker/pin-memory settings.

## 4. Smoke Test

After preprocessing and (optionally) tokenizer training, run:

```
python scripts/debug_deepfashion_dataloader.py ^
  --root ./data/DeepFashion-MultiModal ^
  --batch-size 2 ^
  --tokenizer-path ./data/DeepFashion-MultiModal/tokenizer/spm.model
```

This prints tensor shapes, a few sample captions, and image IDs from each split to verify that everything loads correctly before integrating with your models.

