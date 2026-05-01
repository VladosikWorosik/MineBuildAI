from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .block_model import BlockTransformer, encode_prompt, pad_batch, resolve_device, sample_token, split_dataset
from .block_tokenizer import BLOCKS_TOKEN, SIZE_TOKEN, BlockVocab, write_sponge_schem
from .coord_tokenizer import (
    build_coord_vocab,
    coord_tokens_to_schematic,
    iter_coord_schematics,
    schematic_to_coord_tokens,
)


@dataclass(frozen=True)
class CoordModelConfig:
    max_prompt_tokens: int = 256
    max_blocks: int = 4096
    max_non_air: int = 1024
    min_non_air: int = 1
    max_target_tokens: int = 4104
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1


class CoordSeq2SeqDataset(Dataset):
    def __init__(self, manifest_path: Path, vocab: BlockVocab, config: CoordModelConfig, vanilla_only: bool = False) -> None:
        self.vocab = vocab
        self.config = config
        self.samples = []
        for record, schematic in iter_coord_schematics(
            manifest_path,
            max_blocks=config.max_blocks,
            max_non_air=config.max_non_air,
            min_non_air=config.min_non_air,
            vanilla_only=vanilla_only,
        ):
            prompt = str(record.get("prompt") or Path(str(record.get("path", "build"))).stem)
            target_tokens = schematic_to_coord_tokens(schematic)
            if len(target_tokens) <= config.max_target_tokens:
                self.samples.append((prompt, target_tokens))
        if not self.samples:
            raise ValueError("No coordinate-tokenized schematic records found. Lower --min-non-air or increase --max-non-air/--max-blocks.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt, target_tokens = self.samples[index]
        src = torch.tensor(encode_prompt(prompt, self.config.max_prompt_tokens), dtype=torch.long)
        tgt = torch.tensor(self.vocab.encode(target_tokens), dtype=torch.long)
        return src, tgt


def run_epoch(
    model: BlockTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    target_pad_id: int,
    grad_clip: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(ignore_index=target_pad_id)
    total = 0.0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for src, tgt in tqdm(loader, disable=len(loader) < 2):
            src = src.to(device)
            tgt = tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, model.target_vocab_size), tgt[:, 1:].reshape(-1))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
    return total / max(1, batches)


def save_checkpoint(path: Path, model: BlockTransformer, config: CoordModelConfig, history: list[dict]) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "target_vocab_size": model.target_vocab_size,
            "target_pad_id": model.target_pad_id,
            "history": history,
            "tokenizer": "coord-v1",
        },
        path,
    )


def train_coord_model(
    manifest_path: Path,
    model_dir: Path,
    config: CoordModelConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    val_fraction: float,
    seed: int,
    grad_clip: float,
    vanilla_only: bool = False,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = model_dir / "vocab.json"
    sample_count = build_coord_vocab(
        manifest_path,
        vocab_path,
        max_blocks=config.max_blocks,
        max_non_air=config.max_non_air,
        min_non_air=config.min_non_air,
        vanilla_only=vanilla_only,
    )
    vocab = BlockVocab.load(vocab_path)
    dataset = CoordSeq2SeqDataset(manifest_path, vocab, config, vanilla_only=vanilla_only)
    train_dataset, val_dataset = split_dataset(dataset, val_fraction, seed)
    collate = lambda batch: pad_batch(batch, vocab.pad_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate) if val_dataset else None

    model = BlockTransformer(config, target_vocab_size=len(vocab), target_pad_id=vocab.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, vocab.pad_id, grad_clip)
        val_loss = run_epoch(model, val_loader, device, None, vocab.pad_id, grad_clip) if val_loader else None
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        score = val_loss if val_loss is not None else train_loss
        if score <= best_loss:
            best_loss = score
            save_checkpoint(model_dir / "coord_model.pt", model, config, history)
    save_checkpoint(model_dir / "coord_last.pt", model, config, history)
    (model_dir / "coord_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return {"records": len(dataset), "schematics": sample_count, "vocab": len(vocab), "best_loss": best_loss}


def load_coord_model(model_dir: Path, device: torch.device, checkpoint_name: str) -> tuple[BlockTransformer, BlockVocab]:
    checkpoint = torch.load(model_dir / checkpoint_name, map_location=device)
    config = CoordModelConfig(**checkpoint["config"])
    vocab = BlockVocab.load(model_dir / "vocab.json")
    model = BlockTransformer(config, target_vocab_size=len(vocab), target_pad_id=vocab.pad_id).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab


def _token_int(token: str, prefix: str) -> int | None:
    if not token.startswith(prefix):
        return None
    try:
        return int(token.split(":", 1)[1])
    except ValueError:
        return None


def generated_dimensions(vocab: BlockVocab, generated: list[int]) -> tuple[int | None, int | None, int | None]:
    tokens = vocab.decode(generated)
    width = _token_int(tokens[2], "W:") if len(tokens) > 2 else None
    height = _token_int(tokens[3], "H:") if len(tokens) > 3 else None
    length = _token_int(tokens[4], "L:") if len(tokens) > 4 else None
    return width, height, length


def prefixed_ids(vocab: BlockVocab, prefix: str, upper_bound: int | None = None) -> set[int]:
    ids: set[int] = set()
    for token, idx in vocab.token_to_id.items():
        if not token.startswith(prefix):
            continue
        if upper_bound is not None:
            value = _token_int(token, prefix)
            if value is None or value < 0 or value >= upper_bound:
                continue
        ids.add(idx)
    return ids


def allowed_coord_token_ids(vocab: BlockVocab, generated: list[int], min_non_air: int) -> set[int]:
    token_count = len(generated)
    if token_count == 1:
        return {vocab.token_to_id[SIZE_TOKEN]}
    if token_count == 2:
        return {idx for token, idx in vocab.token_to_id.items() if token.startswith("W:")}
    if token_count == 3:
        return {idx for token, idx in vocab.token_to_id.items() if token.startswith("H:")}
    if token_count == 4:
        return {idx for token, idx in vocab.token_to_id.items() if token.startswith("L:")}
    if token_count == 5:
        return {vocab.token_to_id[BLOCKS_TOKEN]}

    width, height, length = generated_dimensions(vocab, generated)
    coord_position = (token_count - 6) % 4
    completed_blocks = max(0, (token_count - 6) // 4)
    if coord_position == 0:
        allowed = prefixed_ids(vocab, "X:", width)
        if completed_blocks >= min_non_air:
            allowed.add(vocab.eos_id)
        return allowed
    if coord_position == 1:
        return prefixed_ids(vocab, "Y:", height)
    if coord_position == 2:
        return prefixed_ids(vocab, "Z:", length)
    return prefixed_ids(vocab, "B:")


def constrained_sample_token(
    logits: torch.Tensor,
    vocab: BlockVocab,
    generated: list[int],
    min_non_air: int,
    temperature: float,
    top_k: int,
) -> int:
    allowed = allowed_coord_token_ids(vocab, generated, min_non_air)
    if not allowed:
        return sample_token(logits, temperature, top_k, banned_ids={vocab.pad_id, vocab.unk_id})
    masked = torch.full_like(logits, float("-inf"))
    allowed_tensor = torch.tensor(sorted(allowed), dtype=torch.long, device=logits.device)
    masked[allowed_tensor] = logits[allowed_tensor]
    return sample_token(masked, temperature, top_k, banned_ids={vocab.pad_id, vocab.unk_id})


def generate_coord_schematic(
    model_dir: Path,
    prompt: str,
    out_path: Path,
    device_name: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    checkpoint_name: str = "coord_model.pt",
) -> dict:
    device = resolve_device(device_name)
    model, vocab = load_coord_model(model_dir, device, checkpoint_name)
    max_tokens = min(max_tokens, model.config.max_target_tokens)
    src = torch.tensor([encode_prompt(prompt, model.config.max_prompt_tokens)], dtype=torch.long, device=device)
    generated = [vocab.bos_id]
    with torch.no_grad():
        for _ in tqdm(range(max_tokens - 1), disable=max_tokens < 1024):
            tgt = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(src, tgt)[0, -1]
            token = constrained_sample_token(
                logits,
                vocab,
                generated,
                min_non_air=model.config.min_non_air,
                temperature=temperature,
                top_k=top_k,
            )
            generated.append(token)
            if token == vocab.eos_id:
                break
    tokens = vocab.decode(generated)
    schematic = coord_tokens_to_schematic(tokens, max_blocks=model.config.max_blocks)
    write_sponge_schem(out_path, schematic)
    non_air = sum(1 for block in schematic.blocks if block != "minecraft:air")
    return {
        "out": str(out_path),
        "tokens": len(generated),
        "blocks": schematic.volume,
        "non_air": non_air,
        "size": [schematic.width, schematic.height, schematic.length],
    }


def config_from_args(args: argparse.Namespace) -> CoordModelConfig:
    max_target_tokens = args.max_target_tokens or (args.max_non_air * 4 + 8)
    return CoordModelConfig(
        max_prompt_tokens=args.max_prompt_tokens,
        max_blocks=args.max_blocks,
        max_non_air=args.max_non_air,
        min_non_air=args.min_non_air,
        max_target_tokens=max_target_tokens,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
