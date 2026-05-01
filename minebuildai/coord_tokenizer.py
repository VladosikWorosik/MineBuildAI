from __future__ import annotations

import argparse
from pathlib import Path

from .block_tokenizer import (
    AIR,
    BLOCKS_TOKEN,
    BOS_TOKEN,
    BlockSchematic,
    BlockVocab,
    EOS_TOKEN,
    PAD_TOKEN,
    SIZE_TOKEN,
    SPECIAL_TOKENS,
    block_token,
    build_block_vocab,
    dim_token,
    is_vanilla_schematic,
    iter_block_schematics,
)


def coord_token(axis: str, value: int) -> str:
    return f"{axis}:{value}"


def non_air_blocks(schematic: BlockSchematic) -> list[tuple[int, int, int, str]]:
    blocks: list[tuple[int, int, int, str]] = []
    for index, block in enumerate(schematic.blocks):
        if block == AIR:
            continue
        x = index % schematic.width
        z = (index // schematic.width) % schematic.length
        y = index // (schematic.width * schematic.length)
        blocks.append((x, y, z, block))
    return blocks


def schematic_to_coord_tokens(schematic: BlockSchematic) -> list[str]:
    tokens = [
        BOS_TOKEN,
        SIZE_TOKEN,
        dim_token("W", schematic.width),
        dim_token("H", schematic.height),
        dim_token("L", schematic.length),
        BLOCKS_TOKEN,
    ]
    for x, y, z, block in non_air_blocks(schematic):
        tokens.extend([coord_token("X", x), coord_token("Y", y), coord_token("Z", z), block_token(block)])
    tokens.append(EOS_TOKEN)
    return tokens


def _read_int_token(token: str, prefix: str) -> int | None:
    if not token.startswith(prefix):
        return None
    try:
        return int(token.split(":", 1)[1])
    except ValueError:
        return None


def coord_tokens_to_schematic(tokens: list[str], max_blocks: int) -> BlockSchematic:
    useful: list[str] = []
    for token in tokens:
        if token in {BOS_TOKEN, PAD_TOKEN}:
            continue
        if token == EOS_TOKEN:
            break
        useful.append(token)

    width = height = length = None
    blocks_start = 0
    for index, token in enumerate(useful):
        if token.startswith("W:"):
            width = _read_int_token(token, "W:")
        elif token.startswith("H:"):
            height = _read_int_token(token, "H:")
        elif token.startswith("L:"):
            length = _read_int_token(token, "L:")
        elif token == BLOCKS_TOKEN:
            blocks_start = index + 1
            break

    width = width or 8
    height = height or 8
    length = length or 8
    volume = width * height * length
    if volume <= 0 or volume > max_blocks:
        width = min(max(1, width), 32)
        height = min(max(1, height), 32)
        length = min(max(1, length), max(1, max_blocks // (width * height)))
        volume = width * height * length

    blocks = [AIR] * volume
    pending_x: int | None = None
    pending_y: int | None = None
    pending_z: int | None = None
    for token in useful[blocks_start:]:
        if token.startswith("X:"):
            pending_x = _read_int_token(token, "X:")
        elif token.startswith("Y:"):
            pending_y = _read_int_token(token, "Y:")
        elif token.startswith("Z:"):
            pending_z = _read_int_token(token, "Z:")
        elif token.startswith("B:") and pending_x is not None and pending_y is not None and pending_z is not None:
            if 0 <= pending_x < width and 0 <= pending_y < height and 0 <= pending_z < length:
                index = pending_y * width * length + pending_z * width + pending_x
                blocks[index] = token[2:]
            pending_x = pending_y = pending_z = None
    return BlockSchematic(width=width, height=height, length=length, blocks=blocks)


def iter_coord_schematics(
    manifest_path: Path,
    max_blocks: int,
    max_non_air: int,
    min_non_air: int = 1,
    vanilla_only: bool = False,
) -> list[tuple[dict, BlockSchematic]]:
    samples = []
    for record, schematic in iter_block_schematics(manifest_path, max_blocks=max_blocks, vanilla_only=vanilla_only):
        if vanilla_only and not is_vanilla_schematic(schematic):
            continue
        count = len(non_air_blocks(schematic))
        if min_non_air <= count <= max_non_air:
            samples.append((record, schematic))
    return samples


def build_coord_vocab(
    manifest_path: Path,
    out_path: Path,
    max_blocks: int,
    max_non_air: int,
    min_non_air: int = 1,
    vanilla_only: bool = False,
) -> int:
    tokens = set(SPECIAL_TOKENS)
    samples = iter_coord_schematics(
        manifest_path,
        max_blocks=max_blocks,
        max_non_air=max_non_air,
        min_non_air=min_non_air,
        vanilla_only=vanilla_only,
    )
    for _record, schematic in samples:
        tokens.update(schematic_to_coord_tokens(schematic))
    ordered = SPECIAL_TOKENS + sorted(token for token in tokens if token not in SPECIAL_TOKENS)
    BlockVocab({token: index for index, token in enumerate(ordered)}).save(out_path)
    return len(samples)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a coordinate-token vocabulary from schematic files.")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--out", default="models/coord/vocab.json")
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--max-non-air", type=int, default=1024)
    parser.add_argument("--min-non-air", type=int, default=1)
    parser.add_argument("--vanilla-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    count = build_coord_vocab(
        Path(args.manifest),
        Path(args.out),
        max_blocks=args.max_blocks,
        max_non_air=args.max_non_air,
        min_non_air=args.min_non_air,
        vanilla_only=args.vanilla_only,
    )
    print(f"Built coordinate vocabulary from {count} schematics: {args.out}")


if __name__ == "__main__":
    main()
