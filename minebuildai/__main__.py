from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_USER_AGENT = "MineBuildAIDatasetBot/0.1 (+https://example.invalid/minebuildai)"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="minebuildai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", aliases=["parse"], help="Download schematic files from allowed sites.")
    scrape_parser.add_argument("--seed", action="append", required=True)
    scrape_parser.add_argument("--out", default="data/raw")
    scrape_parser.add_argument("--manifest", default="data/manifest.jsonl")
    scrape_parser.add_argument("--allowed-domain", action="append", default=[])
    scrape_parser.add_argument("--max-pages", type=int, default=1000)
    scrape_parser.add_argument("--concurrency", type=int, default=16)
    scrape_parser.add_argument("--timeout", type=float, default=30.0)
    scrape_parser.add_argument("--per-host-delay", type=float, default=0.1)
    scrape_parser.add_argument("--max-file-mb", type=int, default=256)
    scrape_parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    scrape_parser.add_argument("--ignore-robots", action="store_true", help="Disable robots.txt checks. Only use where you have permission.")

    manifest_parser = subparsers.add_parser("manifest", help="Build JSONL manifest from a local schematic directory.")
    manifest_parser.add_argument("--schematics-dir", default="data/raw")
    manifest_parser.add_argument("--out", default="data/manifest.jsonl")

    train_parser = subparsers.add_parser("train-retriever", help="Train retrieval baseline.")
    train_parser.add_argument("--manifest", default="data/manifest.jsonl")
    train_parser.add_argument("--model-dir", default="models/retriever")

    generate_parser = subparsers.add_parser("generate", help="Return nearest schematic for a prompt.")
    generate_parser.add_argument("--model-dir", default="models/retriever")
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--out", required=True)

    neural_train_parser = subparsers.add_parser(
        "train-neural",
        aliases=["train-transformer"],
        help="Train a PyTorch Transformer prompt-to-schematic model.",
    )
    neural_train_parser.add_argument("--manifest", default="data/manifest.jsonl")
    neural_train_parser.add_argument("--model-dir", default="models/neural")
    neural_train_parser.add_argument("--epochs", type=int, default=10)
    neural_train_parser.add_argument("--batch-size", type=int, default=2)
    neural_train_parser.add_argument("--lr", type=float, default=3e-4)
    neural_train_parser.add_argument("--device", default="auto")
    neural_train_parser.add_argument("--max-prompt-tokens", type=int, default=256)
    neural_train_parser.add_argument("--max-target-bytes", type=int, default=8192)
    neural_train_parser.add_argument("--truncate-targets", action="store_true")
    neural_train_parser.add_argument("--val-fraction", type=float, default=0.05)
    neural_train_parser.add_argument("--seed", type=int, default=42)
    neural_train_parser.add_argument("--grad-clip", type=float, default=1.0)
    neural_train_parser.add_argument("--d-model", type=int, default=256)
    neural_train_parser.add_argument("--nhead", type=int, default=8)
    neural_train_parser.add_argument("--encoder-layers", type=int, default=4)
    neural_train_parser.add_argument("--decoder-layers", type=int, default=4)
    neural_train_parser.add_argument("--dim-feedforward", type=int, default=1024)
    neural_train_parser.add_argument("--dropout", type=float, default=0.1)

    neural_generate_parser = subparsers.add_parser(
        "generate-neural",
        aliases=["generate-transformer"],
        help="Generate schematic bytes with the PyTorch Transformer.",
    )
    neural_generate_parser.add_argument("--model-dir", default="models/neural")
    neural_generate_parser.add_argument("--prompt", required=True)
    neural_generate_parser.add_argument("--out", required=True)
    neural_generate_parser.add_argument("--device", default="auto")
    neural_generate_parser.add_argument("--max-new-bytes", type=int, default=8192)
    neural_generate_parser.add_argument("--temperature", type=float, default=0.0)
    neural_generate_parser.add_argument("--top-k", type=int, default=50)
    neural_generate_parser.add_argument("--checkpoint", default="model.pt")

    block_vocab_parser = subparsers.add_parser("build-block-vocab", help="Build block-level token vocabulary from Sponge .schem files.")
    block_vocab_parser.add_argument("--manifest", default="data/manifest.jsonl")
    block_vocab_parser.add_argument("--out", default="models/block/vocab.json")
    block_vocab_parser.add_argument("--max-blocks", type=int, default=4096)
    block_vocab_parser.add_argument("--vanilla-only", action="store_true")

    block_train_parser = subparsers.add_parser("train-block", help="Train a block-token Transformer and save a .schem generator.")
    block_train_parser.add_argument("--manifest", default="data/manifest.jsonl")
    block_train_parser.add_argument("--model-dir", default="models/block")
    block_train_parser.add_argument("--epochs", type=int, default=10)
    block_train_parser.add_argument("--batch-size", type=int, default=2)
    block_train_parser.add_argument("--lr", type=float, default=3e-4)
    block_train_parser.add_argument("--device", default="auto")
    block_train_parser.add_argument("--max-prompt-tokens", type=int, default=256)
    block_train_parser.add_argument("--max-blocks", type=int, default=4096)
    block_train_parser.add_argument("--max-target-tokens", type=int, default=None)
    block_train_parser.add_argument("--val-fraction", type=float, default=0.05)
    block_train_parser.add_argument("--seed", type=int, default=42)
    block_train_parser.add_argument("--grad-clip", type=float, default=1.0)
    block_train_parser.add_argument("--d-model", type=int, default=256)
    block_train_parser.add_argument("--nhead", type=int, default=8)
    block_train_parser.add_argument("--encoder-layers", type=int, default=4)
    block_train_parser.add_argument("--decoder-layers", type=int, default=4)
    block_train_parser.add_argument("--dim-feedforward", type=int, default=1024)
    block_train_parser.add_argument("--dropout", type=float, default=0.1)
    block_train_parser.add_argument("--vanilla-only", action="store_true")
    block_train_parser.add_argument("--air-loss-weight", type=float, default=1.0)

    block_generate_parser = subparsers.add_parser("generate-block", help="Generate a valid Sponge .schem with the block-token Transformer.")
    block_generate_parser.add_argument("--model-dir", default="models/block")
    block_generate_parser.add_argument("--prompt", required=True)
    block_generate_parser.add_argument("--out", required=True)
    block_generate_parser.add_argument("--device", default="auto")
    block_generate_parser.add_argument("--max-tokens", type=int, default=4104)
    block_generate_parser.add_argument("--temperature", type=float, default=0.0)
    block_generate_parser.add_argument("--top-k", type=int, default=50)
    block_generate_parser.add_argument("--checkpoint", default="block_model.pt")

    coord_vocab_parser = subparsers.add_parser("build-coord-vocab", help="Build coordinate-token vocabulary from non-air schematic blocks.")
    coord_vocab_parser.add_argument("--manifest", default="data/manifest.jsonl")
    coord_vocab_parser.add_argument("--out", default="models/coord/vocab.json")
    coord_vocab_parser.add_argument("--max-blocks", type=int, default=4096)
    coord_vocab_parser.add_argument("--max-non-air", type=int, default=1024)
    coord_vocab_parser.add_argument("--min-non-air", type=int, default=1)
    coord_vocab_parser.add_argument("--vanilla-only", action="store_true")

    coord_train_parser = subparsers.add_parser("train-coord", help="Train coordinate-token Transformer using only non-air blocks.")
    coord_train_parser.add_argument("--manifest", default="data/manifest.jsonl")
    coord_train_parser.add_argument("--model-dir", default="models/coord")
    coord_train_parser.add_argument("--epochs", type=int, default=10)
    coord_train_parser.add_argument("--batch-size", type=int, default=2)
    coord_train_parser.add_argument("--lr", type=float, default=3e-4)
    coord_train_parser.add_argument("--device", default="auto")
    coord_train_parser.add_argument("--max-prompt-tokens", type=int, default=256)
    coord_train_parser.add_argument("--max-blocks", type=int, default=4096)
    coord_train_parser.add_argument("--max-non-air", type=int, default=1024)
    coord_train_parser.add_argument("--min-non-air", type=int, default=1)
    coord_train_parser.add_argument("--max-target-tokens", type=int, default=None)
    coord_train_parser.add_argument("--val-fraction", type=float, default=0.05)
    coord_train_parser.add_argument("--seed", type=int, default=42)
    coord_train_parser.add_argument("--grad-clip", type=float, default=1.0)
    coord_train_parser.add_argument("--d-model", type=int, default=256)
    coord_train_parser.add_argument("--nhead", type=int, default=8)
    coord_train_parser.add_argument("--encoder-layers", type=int, default=4)
    coord_train_parser.add_argument("--decoder-layers", type=int, default=4)
    coord_train_parser.add_argument("--dim-feedforward", type=int, default=1024)
    coord_train_parser.add_argument("--dropout", type=float, default=0.1)
    coord_train_parser.add_argument("--vanilla-only", action="store_true")

    coord_generate_parser = subparsers.add_parser("generate-coord", help="Generate a Sponge .schem with the coordinate-token Transformer.")
    coord_generate_parser.add_argument("--model-dir", default="models/coord")
    coord_generate_parser.add_argument("--prompt", required=True)
    coord_generate_parser.add_argument("--out", required=True)
    coord_generate_parser.add_argument("--device", default="auto")
    coord_generate_parser.add_argument("--max-tokens", type=int, default=4104)
    coord_generate_parser.add_argument("--temperature", type=float, default=0.8)
    coord_generate_parser.add_argument("--top-k", type=int, default=30)
    coord_generate_parser.add_argument("--checkpoint", default="coord_model.pt")

    args = parser.parse_args(argv)
    if args.command in {"scrape", "parse"}:
        import asyncio

        from .scrape import config_from_args, scrape

        stats = asyncio.run(scrape(config_from_args(args)))
        print(
            "Downloaded "
            f"{stats.files_downloaded}/{stats.files_seen} files, "
            f"fetched {stats.pages_fetched}/{stats.pages_seen} pages, "
            f"skipped {stats.files_skipped}, failures {stats.failures}."
        )
    elif args.command == "manifest":
        from .dataset import build_manifest

        count = build_manifest(Path(args.schematics_dir), Path(args.out))
        print(f"Wrote {count} records to {args.out}")
    elif args.command == "train-retriever":
        from .retriever import train_retriever

        count = train_retriever(Path(args.manifest), Path(args.model_dir))
        print(f"Trained retrieval baseline on {count} records.")
    elif args.command == "generate":
        from .retriever import generate_from_prompt

        result = generate_from_prompt(Path(args.model_dir), args.prompt, Path(args.out))
        print(f"Wrote {result['out']} with score {result['score']:.4f}")
    elif args.command in {"train-neural", "train-transformer"}:
        from .neural import config_from_args, train_neural

        result = train_neural(
            manifest_path=Path(args.manifest),
            model_dir=Path(args.model_dir),
            config=config_from_args(args),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device_name=args.device,
            truncate_targets=args.truncate_targets,
            val_fraction=args.val_fraction,
            seed=args.seed,
            grad_clip=args.grad_clip,
        )
        print(f"Trained neural model on {result['records']} records. Best loss: {result['best_loss']:.4f}")
    elif args.command in {"generate-neural", "generate-transformer"}:
        from .neural import generate_neural

        result = generate_neural(
            model_dir=Path(args.model_dir),
            prompt=args.prompt,
            out_path=Path(args.out),
            device_name=args.device,
            max_new_bytes=args.max_new_bytes,
            temperature=args.temperature,
            top_k=args.top_k,
            checkpoint_name=args.checkpoint,
        )
        print(f"Wrote {result['out']} ({result['bytes']} bytes, {result['tokens']} generated tokens).")
    elif args.command == "build-block-vocab":
        from .block_tokenizer import build_block_vocab

        count = build_block_vocab(Path(args.manifest), Path(args.out), args.max_blocks, vanilla_only=args.vanilla_only)
        print(f"Built block vocabulary from {count} schematics: {args.out}")
    elif args.command == "train-block":
        from .block_model import config_from_args, train_block_model

        result = train_block_model(
            manifest_path=Path(args.manifest),
            model_dir=Path(args.model_dir),
            config=config_from_args(args),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device_name=args.device,
            val_fraction=args.val_fraction,
            seed=args.seed,
            grad_clip=args.grad_clip,
            vanilla_only=args.vanilla_only,
            air_loss_weight=args.air_loss_weight,
        )
        print(
            f"Trained block model on {result['records']} records, "
            f"vocab {result['vocab']}. Best loss: {result['best_loss']:.4f}"
        )
    elif args.command == "generate-block":
        from .block_model import generate_block_schematic

        result = generate_block_schematic(
            model_dir=Path(args.model_dir),
            prompt=args.prompt,
            out_path=Path(args.out),
            device_name=args.device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            checkpoint_name=args.checkpoint,
        )
        print(f"Wrote {result['out']} size={result['size']} blocks={result['blocks']} tokens={result['tokens']}.")
    elif args.command == "build-coord-vocab":
        from .coord_tokenizer import build_coord_vocab

        count = build_coord_vocab(
            Path(args.manifest),
            Path(args.out),
            max_blocks=args.max_blocks,
            max_non_air=args.max_non_air,
            min_non_air=args.min_non_air,
            vanilla_only=args.vanilla_only,
        )
        print(f"Built coordinate vocabulary from {count} schematics: {args.out}")
    elif args.command == "train-coord":
        from .coord_model import config_from_args, train_coord_model

        result = train_coord_model(
            manifest_path=Path(args.manifest),
            model_dir=Path(args.model_dir),
            config=config_from_args(args),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device_name=args.device,
            val_fraction=args.val_fraction,
            seed=args.seed,
            grad_clip=args.grad_clip,
            vanilla_only=args.vanilla_only,
        )
        print(
            f"Trained coordinate model on {result['records']} records, "
            f"vocab {result['vocab']}. Best loss: {result['best_loss']:.4f}"
        )
    elif args.command == "generate-coord":
        from .coord_model import generate_coord_schematic

        result = generate_coord_schematic(
            model_dir=Path(args.model_dir),
            prompt=args.prompt,
            out_path=Path(args.out),
            device_name=args.device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            checkpoint_name=args.checkpoint,
        )
        print(
            f"Wrote {result['out']} size={result['size']} blocks={result['blocks']} "
            f"non_air={result['non_air']} tokens={result['tokens']}."
        )


if __name__ == "__main__":
    main()
