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

from .dataset import read_jsonl


PAD = 0
BOS = 1
EOS = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 256 + BYTE_OFFSET


@dataclass(frozen=True)
class NeuralConfig:
    max_prompt_tokens: int = 256
    max_target_bytes: int = 8192
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1


def encode_bytes(value: bytes, max_tokens: int) -> list[int]:
    tokens = [BOS]
    tokens.extend(byte + BYTE_OFFSET for byte in value[: max(0, max_tokens - 2)])
    tokens.append(EOS)
    return tokens


def encode_prompt(prompt: str, max_tokens: int) -> list[int]:
    return encode_bytes(prompt.encode("utf-8", errors="replace"), max_tokens)


def decode_bytes(tokens: list[int]) -> bytes:
    output = bytearray()
    for token in tokens:
        if token == EOS:
            break
        if token >= BYTE_OFFSET:
            output.append(token - BYTE_OFFSET)
    return bytes(output)


class SchematicSeq2SeqDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        max_prompt_tokens: int,
        max_target_bytes: int,
        truncate_targets: bool = False,
    ) -> None:
        self.records: list[dict] = []
        for record in read_jsonl(manifest_path):
            prompt = str(record.get("prompt") or "").strip()
            path_value = record.get("path")
            if not prompt or not path_value:
                continue
            path = Path(path_value)
            if not path.exists() or not path.is_file():
                continue
            size = path.stat().st_size
            if size > max_target_bytes and not truncate_targets:
                continue
            self.records.append({"prompt": prompt, "path": str(path)})

        if not self.records:
            raise ValueError(
                "No trainable records found. Use smaller schematics, increase --max-target-bytes, "
                "or pass --truncate-targets."
            )
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_bytes = max_target_bytes

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        src = encode_prompt(record["prompt"], self.max_prompt_tokens)
        target_bytes = Path(record["path"]).read_bytes()
        tgt = encode_bytes(target_bytes, self.max_target_bytes + 2)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def pad_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    src_items, tgt_items = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src_items, batch_first=True, padding_value=PAD)
    tgt = nn.utils.rnn.pad_sequence(tgt_items, batch_first=True, padding_value=PAD)
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(tokens + self.pe[:, : tokens.size(1)])


class SchematicTransformer(nn.Module):
    def __init__(self, config: NeuralConfig) -> None:
        super().__init__()
        self.config = config
        max_len = max(config.max_prompt_tokens, config.max_target_bytes + 2) + 8
        self.embedding = nn.Embedding(VOCAB_SIZE, config.d_model, padding_idx=PAD)
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
        self.output = nn.Linear(config.d_model, VOCAB_SIZE)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_padding_mask = src.eq(PAD)
        tgt_padding_mask = tgt_in.eq(PAD)
        tgt_mask = causal_mask(tgt_in.size(1), tgt_in.device)
        src_emb = self.positional(self.embedding(src) * math.sqrt(self.config.d_model))
        tgt_emb = self.positional(self.embedding(tgt_in) * math.sqrt(self.config.d_model))
        hidden = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output(hidden)


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)


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
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset


def run_epoch(
    model: SchematicTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    total_loss = 0.0
    total_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for src, tgt in tqdm(loader, disable=len(loader) < 2):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
    return total_loss / max(1, total_batches)


def train_neural(
    manifest_path: Path,
    model_dir: Path,
    config: NeuralConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    truncate_targets: bool,
    val_fraction: float,
    seed: int,
    grad_clip: float,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    dataset = SchematicSeq2SeqDataset(
        manifest_path=manifest_path,
        max_prompt_tokens=config.max_prompt_tokens,
        max_target_bytes=config.max_target_bytes,
        truncate_targets=truncate_targets,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_fraction, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_batch) if val_dataset else None

    model = SchematicTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    best_val_loss = float("inf")
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, grad_clip)
        val_loss = run_epoch(model, val_loader, device, None, grad_clip) if val_loader else None
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        score = val_loss if val_loss is not None else train_loss
        if score <= best_val_loss:
            best_val_loss = score
            save_checkpoint(model_dir / "model.pt", model, config, history)

    save_checkpoint(model_dir / "last.pt", model, config, history)
    (model_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return {"records": len(dataset), "model_dir": str(model_dir), "best_loss": best_val_loss}


def save_checkpoint(path: Path, model: SchematicTransformer, config: NeuralConfig, history: list[dict]) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "history": history,
            "tokens": {"pad": PAD, "bos": BOS, "eos": EOS, "byte_offset": BYTE_OFFSET, "vocab_size": VOCAB_SIZE},
        },
        path,
    )


def load_model(model_dir: Path, device: torch.device, checkpoint_name: str = "model.pt") -> SchematicTransformer:
    checkpoint_path = model_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = NeuralConfig(**checkpoint["config"])
    model = SchematicTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits.clone()
    logits[PAD] = float("-inf")
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


def generate_neural(
    model_dir: Path,
    prompt: str,
    out_path: Path,
    device_name: str,
    max_new_bytes: int,
    temperature: float,
    top_k: int,
    checkpoint_name: str = "model.pt",
) -> dict:
    device = resolve_device(device_name)
    model = load_model(model_dir, device, checkpoint_name)
    if max_new_bytes > model.config.max_target_bytes:
        raise ValueError(
            f"--max-new-bytes={max_new_bytes} exceeds the trained limit "
            f"of {model.config.max_target_bytes}. Retrain with a larger --max-target-bytes."
        )
    src_tokens = encode_prompt(prompt, model.config.max_prompt_tokens)
    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    generated = [BOS]

    with torch.no_grad():
        for _ in tqdm(range(max_new_bytes + 1), disable=max_new_bytes < 1024):
            tgt = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(src, tgt)[0, -1]
            token = sample_token(logits, temperature=temperature, top_k=top_k)
            if token == EOS:
                break
            generated.append(token)

    output = decode_bytes(generated[1:])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(output)
    return {"out": str(out_path), "bytes": len(output), "tokens": len(generated) - 1}


def config_from_args(args: argparse.Namespace) -> NeuralConfig:
    return NeuralConfig(
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_bytes=args.max_target_bytes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
