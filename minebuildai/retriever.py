from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .dataset import read_jsonl


MODEL_FILENAME = "retriever.joblib"


def train_retriever(manifest_path: Path, model_dir: Path) -> int:
    records = read_jsonl(manifest_path)
    records = [record for record in records if record.get("prompt") and record.get("path")]
    if not records:
        raise ValueError(f"No usable records found in {manifest_path}")

    prompts = [str(record["prompt"]) for record in records]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    matrix = vectorizer.fit_transform(prompts)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "records": records}, model_dir / MODEL_FILENAME)
    return len(records)


def generate_from_prompt(model_dir: Path, prompt: str, out_path: Path) -> dict:
    model = joblib.load(model_dir / MODEL_FILENAME)
    vectorizer: TfidfVectorizer = model["vectorizer"]
    matrix = model["matrix"]
    records: list[dict] = model["records"]

    query = vectorizer.transform([prompt])
    scores = cosine_similarity(query, matrix)[0]
    best_index = int(scores.argmax())
    best_record = records[best_index]
    source_path = Path(best_record["path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Schematic from manifest is missing: {source_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, out_path)
    return {"score": float(scores[best_index]), "record": best_record, "out": str(out_path)}


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prompt-to-schematic retrieval baseline.")
    parser.add_argument("--manifest", default="data/manifest.jsonl")
    parser.add_argument("--model-dir", default="models/retriever")
    return parser.parse_args(argv)


def parse_generate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a schematic by retrieving nearest dataset item.")
    parser.add_argument("--model-dir", default="models/retriever")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def train_main(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)
    count = train_retriever(Path(args.manifest), Path(args.model_dir))
    print(f"Trained retrieval baseline on {count} records: {Path(args.model_dir) / MODEL_FILENAME}")


def generate_main(argv: list[str] | None = None) -> None:
    args = parse_generate_args(argv)
    result = generate_from_prompt(Path(args.model_dir), args.prompt, Path(args.out))
    record = result["record"]
    print(f"Wrote {result['out']} from {record.get('path')} with score {result['score']:.4f}")


if __name__ == "__main__":
    train_main()
