from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .dataset import read_jsonl
from .nbt import (
    NbtByteArray,
    NbtIntArray,
    NbtLongArray,
    NbtList,
    TAG_BYTE_ARRAY,
    TAG_COMPOUND,
    TAG_INT,
    TAG_INT_ARRAY,
    TAG_LIST,
    TAG_LONG,
    TAG_SHORT,
    decode_varints,
    encode_varints,
    read_nbt,
    write_gzip_nbt,
)


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<size>", "<blocks>"]
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SIZE_TOKEN = "<size>"
BLOCKS_TOKEN = "<blocks>"
AIR = "minecraft:air"

LEGACY_ID_MAP = {
    0: "minecraft:air",
    1: "minecraft:stone",
    2: "minecraft:grass_block",
    3: "minecraft:dirt",
    4: "minecraft:cobblestone",
    5: "minecraft:oak_planks",
    7: "minecraft:bedrock",
    8: "minecraft:water",
    9: "minecraft:water",
    10: "minecraft:lava",
    11: "minecraft:lava",
    12: "minecraft:sand",
    13: "minecraft:gravel",
    14: "minecraft:gold_ore",
    15: "minecraft:iron_ore",
    16: "minecraft:coal_ore",
    17: "minecraft:oak_log",
    18: "minecraft:oak_leaves",
    20: "minecraft:glass",
    22: "minecraft:lapis_block",
    24: "minecraft:sandstone",
    35: "minecraft:white_wool",
    41: "minecraft:gold_block",
    42: "minecraft:iron_block",
    45: "minecraft:bricks",
    46: "minecraft:tnt",
    47: "minecraft:bookshelf",
    48: "minecraft:mossy_cobblestone",
    49: "minecraft:obsidian",
    50: "minecraft:torch",
    53: "minecraft:oak_stairs",
    54: "minecraft:chest",
    56: "minecraft:diamond_ore",
    57: "minecraft:diamond_block",
    58: "minecraft:crafting_table",
    61: "minecraft:furnace",
    67: "minecraft:cobblestone_stairs",
    79: "minecraft:ice",
    80: "minecraft:snow_block",
    82: "minecraft:clay",
    87: "minecraft:netherrack",
    89: "minecraft:glowstone",
    98: "minecraft:stone_bricks",
    101: "minecraft:iron_bars",
    102: "minecraft:glass_pane",
    103: "minecraft:melon",
    109: "minecraft:stone_brick_stairs",
    112: "minecraft:nether_bricks",
    121: "minecraft:end_stone",
    133: "minecraft:emerald_block",
    152: "minecraft:redstone_block",
    155: "minecraft:quartz_block",
    159: "minecraft:white_terracotta",
    169: "minecraft:sea_lantern",
}


@dataclass(frozen=True)
class BlockSchematic:
    width: int
    height: int
    length: int
    blocks: list[str]
    source_path: str | None = None

    @property
    def volume(self) -> int:
        return self.width * self.height * self.length


class BlockVocab:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.pad_id = token_to_id[PAD_TOKEN]
        self.bos_id = token_to_id[BOS_TOKEN]
        self.eos_id = token_to_id[EOS_TOKEN]
        self.unk_id = token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(int(idx), UNK_TOKEN) for idx in ids]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BlockVocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def load_sponge_schem(path: Path) -> BlockSchematic:
    root = read_nbt(path).value
    required = ["Width", "Height", "Length", "Palette", "BlockData"]
    missing = [key for key in required if key not in root]
    if missing:
        raise ValueError(f"Not a Sponge .schem file, missing: {', '.join(missing)}")

    width = int(root["Width"])
    height = int(root["Height"])
    length = int(root["Length"])
    volume = width * height * length
    palette_raw = root["Palette"]
    if not isinstance(palette_raw, dict):
        raise ValueError("Schematic Palette must be an NBT compound")
    index_to_block = {int(index): str(block) for block, index in palette_raw.items()}
    block_data_raw = root["BlockData"]
    if not isinstance(block_data_raw, NbtByteArray):
        raise ValueError("Schematic BlockData must be a byte array")
    palette_indices = decode_varints(block_data_raw.data, volume)
    blocks = [index_to_block.get(index, AIR) for index in palette_indices]
    return BlockSchematic(width=width, height=height, length=length, blocks=blocks, source_path=str(path))


def load_legacy_schematic(path: Path) -> BlockSchematic:
    root = read_nbt(path).value
    required = ["Width", "Height", "Length", "Blocks"]
    missing = [key for key in required if key not in root]
    if missing:
        raise ValueError(f"Not an MCEdit .schematic file, missing: {', '.join(missing)}")
    width = int(root["Width"])
    height = int(root["Height"])
    length = int(root["Length"])
    volume = width * height * length
    block_ids_raw = root["Blocks"]
    if not isinstance(block_ids_raw, NbtByteArray):
        raise ValueError("Legacy schematic Blocks must be a byte array")
    low_ids = list(block_ids_raw.data)
    if len(low_ids) != volume:
        raise ValueError(f"Legacy schematic block count mismatch: {len(low_ids)} != {volume}")

    high_ids = [0] * volume
    add_blocks = root.get("AddBlocks")
    if isinstance(add_blocks, NbtByteArray):
        for index, value in enumerate(add_blocks.data):
            block_index = index * 2
            if block_index < volume:
                high_ids[block_index] = value & 0x0F
            if block_index + 1 < volume:
                high_ids[block_index + 1] = (value & 0xF0) >> 4

    id_to_block = dict(LEGACY_ID_MAP)
    mapping = root.get("SchematicaMapping")
    if isinstance(mapping, dict):
        for block, block_id in mapping.items():
            id_to_block[int(block_id)] = str(block)

    blocks = []
    for low, high in zip(low_ids, high_ids):
        block_id = low | (high << 8)
        blocks.append(id_to_block.get(block_id, AIR))
    return BlockSchematic(width=width, height=height, length=length, blocks=blocks, source_path=str(path))


def block_state_name(state: dict) -> str:
    name = str(state.get("Name", AIR))
    properties = state.get("Properties")
    if not isinstance(properties, dict) or not properties:
        return name
    props = ",".join(f"{key}={properties[key]}" for key in sorted(properties))
    return f"{name}[{props}]"


def unsigned_long(value: int) -> int:
    return value & ((1 << 64) - 1)


def unpack_litematic_blockstates(longs: list[int], palette_size: int, volume: int) -> list[int]:
    if volume == 0:
        return []
    bits = max(2, math.ceil(math.log2(max(1, palette_size))))
    mask = (1 << bits) - 1
    unsigned = [unsigned_long(value) for value in longs]
    values: list[int] = []
    for index in range(volume):
        bit_index = index * bits
        long_index = bit_index // 64
        bit_offset = bit_index % 64
        if long_index >= len(unsigned):
            values.append(0)
            continue
        value = unsigned[long_index] >> bit_offset
        overflow = bit_offset + bits - 64
        if overflow > 0 and long_index + 1 < len(unsigned):
            value |= unsigned[long_index + 1] << (bits - overflow)
        values.append(value & mask)
    return values


def load_litematic(path: Path) -> BlockSchematic:
    root = read_nbt(path).value
    regions = root.get("Regions")
    if not isinstance(regions, dict) or not regions:
        raise ValueError("Litematic file has no Regions compound")

    region_name, region = next(iter(regions.items()))
    if not isinstance(region, dict):
        raise ValueError(f"Litematic region {region_name} is not a compound")
    size = region.get("Size")
    if not isinstance(size, dict):
        raise ValueError("Litematic region has no Size compound")
    width = abs(int(size.get("x", 0)))
    height = abs(int(size.get("y", 0)))
    length = abs(int(size.get("z", 0)))
    volume = width * height * length

    palette_raw = region.get("BlockStatePalette")
    blockstates_raw = region.get("BlockStates")
    if not isinstance(palette_raw, NbtList) or not isinstance(blockstates_raw, NbtLongArray):
        raise ValueError("Litematic region is missing BlockStatePalette or BlockStates")
    palette = [block_state_name(item) if isinstance(item, dict) else AIR for item in palette_raw.items]
    indices = unpack_litematic_blockstates(blockstates_raw.data, len(palette), volume)
    blocks = [palette[index] if 0 <= index < len(palette) else AIR for index in indices]
    return BlockSchematic(width=width, height=height, length=length, blocks=blocks, source_path=str(path))


def load_block_schematic(path: Path) -> BlockSchematic:
    if path.suffix.lower() == ".schem":
        return load_sponge_schem(path)
    if path.suffix.lower() == ".schematic":
        return load_legacy_schematic(path)
    if path.suffix.lower() == ".litematic":
        return load_litematic(path)
    raise ValueError(f"Unsupported block schematic extension: {path.suffix}")


def write_sponge_schem(path: Path, schematic: BlockSchematic, data_version: int = 3465) -> None:
    if schematic.volume != len(schematic.blocks):
        raise ValueError("Block count does not match schematic dimensions")
    palette: dict[str, int] = {}
    indices: list[int] = []
    for block in schematic.blocks:
        if block not in palette:
            palette[block] = len(palette)
        indices.append(palette[block])
    palette_items = [(block, TAG_INT, idx) for block, idx in sorted(palette.items(), key=lambda item: item[1])]
    root_items = [
        ("Version", TAG_INT, 2),
        ("DataVersion", TAG_INT, data_version),
        ("Width", TAG_SHORT, schematic.width),
        ("Height", TAG_SHORT, schematic.height),
        ("Length", TAG_SHORT, schematic.length),
        ("Offset", TAG_INT_ARRAY, NbtIntArray([0, 0, 0])),
        ("PaletteMax", TAG_INT, len(palette)),
        ("Palette", TAG_COMPOUND, palette_items),
        ("BlockData", TAG_BYTE_ARRAY, NbtByteArray(encode_varints(indices))),
        ("BlockEntities", TAG_LIST, NbtList(TAG_COMPOUND, [])),
        ("Entities", TAG_LIST, NbtList(TAG_COMPOUND, [])),
        ("Metadata", TAG_COMPOUND, [("Date", TAG_LONG, 0)]),
    ]
    write_gzip_nbt(path, "Schematic", root_items)


def dim_token(axis: str, value: int) -> str:
    return f"{axis}:{value}"


def block_token(block: str) -> str:
    return f"B:{block}"


def is_vanilla_block(block: str) -> bool:
    return block == AIR or block.startswith("minecraft:")


def is_vanilla_schematic(schematic: BlockSchematic) -> bool:
    return all(is_vanilla_block(block) for block in schematic.blocks)


def schematic_to_tokens(schematic: BlockSchematic) -> list[str]:
    return [
        BOS_TOKEN,
        SIZE_TOKEN,
        dim_token("W", schematic.width),
        dim_token("H", schematic.height),
        dim_token("L", schematic.length),
        BLOCKS_TOKEN,
        *[block_token(block) for block in schematic.blocks],
        EOS_TOKEN,
    ]


def tokens_to_schematic(tokens: list[str], max_blocks: int) -> BlockSchematic:
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
            width = int(token.split(":", 1)[1])
        elif token.startswith("H:"):
            height = int(token.split(":", 1)[1])
        elif token.startswith("L:"):
            length = int(token.split(":", 1)[1])
        elif token == BLOCKS_TOKEN:
            blocks_start = index + 1
            break
    if width is None or height is None or length is None:
        width = width or 8
        height = height or 8
        length = length or 8
    volume = width * height * length
    if volume <= 0 or volume > max_blocks:
        width = min(max(1, width), 16)
        height = min(max(1, height), 16)
        length = min(max(1, length), max(1, max_blocks // (width * height)))
        volume = width * height * length
    block_values = [token[2:] for token in useful[blocks_start:] if token.startswith("B:")]
    if len(block_values) < volume:
        block_values.extend([AIR] * (volume - len(block_values)))
    return BlockSchematic(width=width, height=height, length=length, blocks=block_values[:volume])


def iter_block_schematics(manifest_path: Path, max_blocks: int, vanilla_only: bool = False) -> list[tuple[dict, BlockSchematic]]:
    samples: list[tuple[dict, BlockSchematic]] = []
    for record in read_jsonl(manifest_path):
        path_value = record.get("path")
        if not path_value:
            continue
        path = Path(path_value)
        if path.suffix.lower() not in {".schem", ".schematic", ".litematic"} or not path.exists():
            continue
        try:
            schematic = load_block_schematic(path)
        except Exception:
            continue
        if schematic.volume > max_blocks:
            continue
        if vanilla_only and not is_vanilla_schematic(schematic):
            continue
        samples.append((record, schematic))
    return samples


def build_block_vocab(manifest_path: Path, out_path: Path, max_blocks: int, vanilla_only: bool = False) -> int:
    tokens = set(SPECIAL_TOKENS)
    samples = iter_block_schematics(manifest_path, max_blocks=max_blocks, vanilla_only=vanilla_only)
    for _record, schematic in samples:
        tokens.update(schematic_to_tokens(schematic))
    ordered = SPECIAL_TOKENS + sorted(token for token in tokens if token not in SPECIAL_TOKENS)
    BlockVocab({token: index for index, token in enumerate(ordered)}).save(out_path)
    return len(samples)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a block-level vocabulary from Sponge .schem files.")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--out", default="models/block/vocab.json")
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument("--vanilla-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    count = build_block_vocab(Path(args.manifest), Path(args.out), args.max_blocks, vanilla_only=args.vanilla_only)
    print(f"Built block vocabulary from {count} schematics: {args.out}")


if __name__ == "__main__":
    main()
