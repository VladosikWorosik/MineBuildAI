# MineBuildAI

MineBuildAI is an experimental neural Minecraft building toolkit. It trains PyTorch Transformer models on Minecraft schematic datasets and generates Sponge `.schem` files from text prompts.

The project focuses on prompt-to-schematic generation for vanilla Minecraft builds using block-level and coordinate-level tokenization.

## Features

- Fast async schematic downloader/crawler.
- Dataset manifest builder for `.schem`, `.schematic`, `.litematic`, `.nbt`, and `.zip` files.
- NBT reader/writer for Minecraft schematic formats.
- Legacy MCEdit `.schematic` reader.
- Sponge `.schem` writer.
- Litematica `.litematic` reader.
- Block-level PyTorch Transformer model.
- Coordinate-token PyTorch Transformer model using `X/Y/Z/BLOCK` tokens.
- Vanilla-only filtering to avoid modded blocks.
- CPU and CUDA training support.

## Model Types

### Block Model

The block model serializes the full schematic volume as a block sequence:

```text
<bos> <size> W:8 H:6 L:8 <blocks> B:minecraft:stone B:minecraft:air ... <eos>
```

This works, but the model often overlearns `minecraft:air` because most schematic volume is empty.

### Coordinate Model

The coordinate model stores only non-air blocks:

```text
<bos> <size> W:8 H:6 L:8 <blocks>
X:0 Y:0 Z:0 B:minecraft:stone
X:1 Y:0 Z:0 B:minecraft:oak_planks
<eos>
```

This is the recommended mode. Empty space is restored automatically as `minecraft:air`.

## Installation

```bash
pip install -e .
```

Check that the CLI works:

```bash
python -m minebuildai --help
```

or:

```bash
minebuildai --help
```

## Dataset

Build a manifest from downloaded schematics:

```bash
python -m minebuildai manifest \
  --schematics-dir data/downloaded \
  --out data/manifest_all.jsonl
```

For a cleaner English-only dataset, use `data/manifest_clean_en.jsonl` if it already exists. It removes paths/prompts containing CJK characters.

Check coordinate-token usable samples:

```bash
python -m minebuildai build-coord-vocab \
  --manifest data/manifest_clean_en.jsonl \
  --out models/coord_test/vocab.json \
  --max-blocks 4096 \
  --max-non-air 1024 \
  --min-non-air 8 \
  --vanilla-only
```

## Training

Recommended coordinate model training command for Google Colab CUDA:

```bash
python -m minebuildai train-coord \
  --manifest data/manifest_clean_en.jsonl \
  --model-dir models/coord_colab \
  --epochs 50 \
  --batch-size 4 \
  --device cuda \
  --max-blocks 4096 \
  --max-non-air 1024 \
  --min-non-air 8 \
  --max-prompt-tokens 128 \
  --d-model 160 \
  --nhead 4 \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --dim-feedforward 384 \
  --val-fraction 0.1 \
  --vanilla-only
```

Smaller CPU-safe command:

```bash
python -m minebuildai train-coord \
  --manifest data/manifest_clean_en.jsonl \
  --model-dir models/coord_cpu \
  --epochs 10 \
  --batch-size 2 \
  --device cpu \
  --max-blocks 1024 \
  --max-non-air 256 \
  --min-non-air 8 \
  --max-prompt-tokens 96 \
  --d-model 96 \
  --nhead 4 \
  --encoder-layers 2 \
  --decoder-layers 2 \
  --dim-feedforward 192 \
  --val-fraction 0.1 \
  --vanilla-only
```

## Generation

Generate a schematic with a trained coordinate model:

```bash
python -m minebuildai generate-coord \
  --model-dir models/coord_colab \
  --prompt "small wooden starter house" \
  --out out/house.schem \
  --device cuda \
  --max-tokens 4104 \
  --temperature 1.0 \
  --top-k 50
```

For more stable outputs:

```bash
--temperature 0.7 --top-k 20
```

For more diverse outputs:

```bash
--temperature 1.0 --top-k 50
```

## Output

Generated files are saved as Sponge `.schem` files and can be loaded with tools such as WorldEdit or Litematica-compatible workflows.

Example output path:

```text
out/house.schem
```

## Notes

- This is research/prototype code, not a production model.
- Generation quality depends heavily on dataset quality.
- More clean vanilla schematics usually improve results more than just increasing epochs.
- Coordinate-token training is preferred over raw block-token training.
- Respect website terms, licenses, and `robots.txt` when downloading schematics.
