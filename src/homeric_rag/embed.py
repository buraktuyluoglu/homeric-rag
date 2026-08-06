"""Dense embeddings with BAAI/bge-small-en-v1.5, run through ONNX Runtime.

Uses the Xenova ONNX export (Xenova/bge-small-en-v1.5) so no PyTorch is
needed — everything runs on CPU via onnxruntime. The embedding recipe
mirrors hybrid-retrieval-lab exactly, per the bge-small-en-v1.5 model
card: CLS-token pooling, L2 normalization, and the retrieval instruction
prefix on the *query* side only. The model revision is pinned so a fresh
clone reproduces the committed vectors bit-for-bit modulo BLAS noise.

Run as a module to embed every chunk in data/chunks.jsonl:

    python -m homeric_rag.embed

Output: data/embeddings.npz — ``ids`` (chunk ids, aligned row-wise) and
``vectors`` (float32, shape (n_chunks, 384), L2-normalized).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"
MODEL_CACHE = Path(os.environ.get("HOMERIC_RAG_MODEL_CACHE", ".cache/hf"))

MODEL_REPO = "Xenova/bge-small-en-v1.5"
# Pinned to the same snapshot hybrid-retrieval-lab recorded its runs with.
MODEL_REVISION = "ea104dacec62c0de699686887e3f920caeb4f3e3"
MODEL_NAME = "bge-small-en-v1.5"
DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32

_session = None
_tokenizer = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            MODEL_REPO, "onnx/model.onnx", revision=MODEL_REVISION, cache_dir=MODEL_CACHE
        )
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _session


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO, revision=MODEL_REVISION, cache_dir=MODEL_CACHE
        )
    return _tokenizer


def _embed_batch(texts: list[str]) -> np.ndarray:
    tokenizer = _get_tokenizer()
    session = _get_session()
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in {i.name for i in session.get_inputs()}:
        inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
    (last_hidden_state,) = session.run(["last_hidden_state"], inputs)
    cls = last_hidden_state[:, 0, :]  # CLS pooling, per the model card
    norms = np.linalg.norm(cls, axis=1, keepdims=True)
    return (cls / norms).astype(np.float32)


def embed_texts(texts: list[str], *, is_query: bool = False) -> np.ndarray:
    """Embed texts in batches. Returns float32 (n, 384), L2-normalized
    row-wise. Queries get the bge retrieval instruction prefix."""
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    batches = [_embed_batch(texts[i : i + BATCH_SIZE]) for i in range(0, len(texts), BATCH_SIZE)]
    return np.vstack(batches) if batches else np.empty((0, DIM), dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string; returns a (384,) float32 vector."""
    return embed_texts([text], is_query=True)[0]


def main(data_dir: Path = DATA_DIR) -> Path:
    chunks = [
        json.loads(line)
        for line in (data_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    started = time.perf_counter()
    vectors = embed_texts([c["text"] for c in chunks])
    elapsed = time.perf_counter() - started

    out_path = data_dir / "embeddings.npz"
    np.savez_compressed(out_path, ids=np.array([c["id"] for c in chunks]), vectors=vectors)
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"embedded {len(chunks)} chunks with {MODEL_NAME} in {elapsed:.1f}s")
    print(f"wrote {out_path} ({vectors.shape[0]}x{vectors.shape[1]} float32, {size_mb:.2f} MB)")
    return out_path


if __name__ == "__main__":
    main()
