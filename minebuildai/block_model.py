from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .block_tokenizer import (
    BlockVocab,
    iter_block_schematics,
    schematic_to_tokens,
    tokens_to_schematic,
    write_sponge_schem,
)


SRC_PAD = 0
SRC_BOS = 1
SRC_EOS = 2
SRC_BYTE_OFFSET = 3
SRC_VOCAB_SIZE = 259


@dataclass(frozen=True)
class BlockModelConfig:
    max_prompt_tokens: int = 256
    max_blocks: int = 4096
    max_target_tokens: int = 4104
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1


def encode_prompt(prompt: str, max_tokens: int) -> list[int]:
    raw = prompt.encode("utf-8", errors="replace")[: max(0, max_tokens - 2)]
    return [SRC_BOS, *[byte + SRC_BYTE_OFFSET for byte in raw], SRC_EOS]


class BlockSeq2SeqDataset(Dataset):
    def __init__(self, manifest_path: Path, vocab: BlockVocab, config: BlockModelConfig, vanilla_only: bool = False) -> None:
        self.vocab = vocab
        self.config = config
        self.samples = []
        for record, schematic in iter_block_schematics(manifest_path, max_blocks=config.max_blocks, vanilla_only=vanilla_only):
            prompt = str(record.get("prompt") or Path(str(record.get("path", "build"))).stem)
            target_tokens = schematic_to_tokens(schematic)
            if len(target_tokens) > config.max_target_tokens:
                continue
            self.samples.append((prompt, target_tokens))
        if not self.samples:
            raise ValueError("No block-tokenized .schem records found. Need Sponge .schem files within --max-blocks.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt, target_tokens = self.samples[index]
        src = torch.tensor(encode_prompt(prompt, self.config.max_prompt_tokens), dtype=torch.long)
        tgt = torch.tensor(self.vocab.encode(target_tokens), dtype=torch.long)
        return src, tgt


def pad_batch(batch: list[tuple[torch.Tensor, torch.Tensor]], target_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    src_items, tgt_items = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src_items, batch_first=True, padding_value=SRC_PAD)
    tgt = nn.utils.rnn.pad_sequence(tgt_items, batch_first=True, padding_value=target_pad)
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.size(1)])


class BlockTransformer(nn.Module):
    def __init__(self, config: BlockModelConfig, target_vocab_size: int, target_pad_id: int) -> None:
        super().__init__()
        self.config = config
        self.target_vocab_size = target_vocab_size
        self.target_pad_id = target_pad_id
        max_len = max(config.max_prompt_tokens, config.max_target_tokens) + 8
        self.src_embedding = nn.Embedding(SRC_VOCAB_SIZE, config.d_model, padding_idx=SRC_PAD)
        self.tgt_embedding = nn.Embedding(target_vocab_size, config.d_model, padding_idx=target_pad_id)
        self.positional = PositionalEncoding(config.d_model, config.dropout, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.output = nn.Linear(config.d_model, target_vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_padding_mask = src.eq(SRC_PAD)
        tgt_padding_mask = tgt_in.eq(self.target_pad_id)
        tgt_mask = torch.triu(torch.ones((tgt_in.size(1), tgt_in.size(1)), dtype=torch.bool, device=tgt_in.device), diagonal=1)
        src_emb = self.positional(self.src_embedding(src) * math.sqrt(self.config.d_model))
        tgt_emb = self.positional(self.tgt_embedding(tgt_in) * math.sqrt(self.config.d_model))
        hidden = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output(hidden)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def split_dataset(dataset: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if val_fraction <= 0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        return dataset, None
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)


def run_epoch(
    model: BlockTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    target_pad_id: int,
    grad_clip: float,
    loss_weight: torch.Tensor | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(ignore_index=target_pad_id, weight=loss_weight)
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


def save_checkpoint(path: Path, model: BlockTransformer, config: BlockModelConfig, history: list[dict]) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "target_vocab_size": model.target_vocab_size,
            "target_pad_id": model.target_pad_id,
            "history": history,
        },
        path,
    )


def train_block_model(
    manifest_path: Path,
    model_dir: Path,
    config: BlockModelConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    val_fraction: float,
    seed: int,
    grad_clip: float,
    vanilla_only: bool = False,
    air_loss_weight: float = 1.0,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    from .block_tokenizer import build_block_vocab

    vocab_path = model_dir / "vocab.json"
    sample_count = build_block_vocab(manifest_path, vocab_path, max_blocks=config.max_blocks, vanilla_only=vanilla_only)
    vocab = BlockVocab.load(vocab_path)
    dataset = BlockSeq2SeqDataset(manifest_path, vocab, config, vanilla_only=vanilla_only)
    train_dataset, val_dataset = split_dataset(dataset, val_fraction, seed)
    collate = lambda batch: pad_batch(batch, vocab.pad_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate) if val_dataset else None

    model = BlockTransformer(config, target_vocab_size=len(vocab), target_pad_id=vocab.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_weight = None
    air_token_id = vocab.token_to_id.get("B:minecraft:air")
    if air_token_id is not None and air_loss_weight != 1.0:
        loss_weight = torch.ones(len(vocab), dtype=torch.float32, device=device)
        loss_weight[air_token_id] = air_loss_weight
    history: list[dict] = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, vocab.pad_id, grad_clip, loss_weight=loss_weight)
        val_loss = run_epoch(model, val_loader, device, None, vocab.pad_id, grad_clip, loss_weight=loss_weight) if val_loader else None
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        score = val_loss if val_loss is not None else train_loss
        if score <= best_loss:
            best_loss = score
            save_checkpoint(model_dir / "block_model.pt", model, config, history)
    save_checkpoint(model_dir / "block_last.pt", model, config, history)
    (model_dir / "block_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return {"records": len(dataset), "schematics": sample_count, "vocab": len(vocab), "best_loss": best_loss}


def load_block_model(model_dir: Path, device: torch.device, checkpoint_name: str) -> tuple[BlockTransformer, BlockVocab]:
    checkpoint = torch.load(model_dir / checkpoint_name, map_location=device)
    config = BlockModelConfig(**checkpoint["config"])
    vocab = BlockVocab.load(model_dir / "vocab.json")
    model = BlockTransformer(config, target_vocab_size=len(vocab), target_pad_id=vocab.pad_id).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, banned_ids: set[int]) -> int:
    logits = logits.clone()
    for idx in banned_ids:
        logits[idx] = float("-inf")
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[int(choice.item())].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_block_schematic(
    model_dir: Path,
    prompt: str,
    out_path: Path,
    device_name: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    checkpoint_name: str = "block_model.pt",
) -> dict:
    device = resolve_device(device_name)
    model, vocab = load_block_model(model_dir, device, checkpoint_name)
    max_tokens = min(max_tokens, model.config.max_target_tokens)
    src = torch.tensor([encode_prompt(prompt, model.config.max_prompt_tokens)], dtype=torch.long, device=device)
    generated = [vocab.bos_id]
    with torch.no_grad():
        for _ in tqdm(range(max_tokens - 1), disable=max_tokens < 1024):
            tgt = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(src, tgt)[0, -1]
            token = sample_token(logits, temperature, top_k, banned_ids={vocab.pad_id, vocab.unk_id})
            generated.append(token)
            if token == vocab.eos_id:
                break
    tokens = vocab.decode(generated)
    schematic = tokens_to_schematic(tokens, max_blocks=model.config.max_blocks)
    write_sponge_schem(out_path, schematic)
    return {"out": str(out_path), "tokens": len(generated), "blocks": schematic.volume, "size": [schematic.width, schematic.height, schematic.length]}


def config_from_args(args: argparse.Namespace) -> BlockModelConfig:
    max_target_tokens = args.max_target_tokens or (args.max_blocks + 8)
    return BlockModelConfig(
        max_prompt_tokens=args.max_prompt_tokens,
        max_blocks=args.max_blocks,
        max_target_tokens=max_target_tokens,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
