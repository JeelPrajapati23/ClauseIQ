"""Step 2a: embed every row of data/{train,val,test}.jsonl with the production embedding
client (app.database.embeddings — Arctic-embed-l-v2.0 via HF's hosted API, same "query: "
prefix as embed_query), caching vectors to data/emb/<sha1(text)>-keyed .npz files.

Batches through embed_documents with the prefix added by hand — what embed_query does for
one text — so ~8.5k rows take ~270 API calls instead of 8.5k. --check verifies batched and
single-call vectors match before trusting the cache. Resumable: cached texts are skipped.

Usage (project venv, from the repo root):
  python Evaluation/offtopic/embed_dataset.py --check
  python Evaluation/offtopic/embed_dataset.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from app.database import embeddings  # noqa: E402
from _keys import text_key  # noqa: E402

DATA = Path(__file__).parent / "data"
CACHE = DATA / "emb" / "cache.npz"
SPLITS = ("train", "val", "test")
BATCH = 32
QUERY_PREFIX = "query: "  # must match _ArcticLegalEmbeddings.embed_query


def load_cache() -> dict[str, np.ndarray]:
    if not CACHE.exists():
        return {}
    z = np.load(CACHE)
    return dict(zip(z["keys"].tolist(), z["vecs"]))


def save_cache(cache: dict[str, np.ndarray]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    keys = list(cache)
    tmp = CACHE.with_name("cache.tmp.npz")
    np.savez(tmp, keys=np.array(keys), vecs=np.stack([cache[k] for k in keys]).astype(np.float32))
    tmp.replace(CACHE)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    for attempt in range(6):
        try:
            return embeddings.embed_documents([QUERY_PREFIX + t for t in texts])
        except Exception as exc:  # HF free tier: transient 429/503s
            if attempt == 5:
                raise
            wait = 10 * 2 ** attempt
            print(f"  embed error ({exc!r:.120}); retrying in {wait}s")
            time.sleep(wait)


def check() -> None:
    samples = ["what is the termination notice period?", "write a poem about autumn", "indemnity"]
    batched = np.array(_embed_batch(samples))
    single = np.array([embeddings.embed_query(t) for t in samples])
    cos = (batched * single).sum(1) / (np.linalg.norm(batched, axis=1) * np.linalg.norm(single, axis=1))
    print(f"dim={batched.shape[1]}  norms={np.linalg.norm(batched, axis=1).round(4)}  "
          f"batched-vs-single cosine={cos.round(6)}")
    if (cos < 0.9999).any():
        raise SystemExit("batched embeddings differ from embed_query — don't use the cache")


def main():
    if "--check" in sys.argv:
        return check()
    cache = load_cache()
    texts = []
    for split in SPLITS:
        texts += [json.loads(line)["text"] for line in (DATA / f"{split}.jsonl").open(encoding="utf-8")]
    todo = list(dict.fromkeys(t for t in texts if text_key(t) not in cache))
    print(f"{len(cache)} cached, {len(todo)} to embed")
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        for t, v in zip(batch, _embed_batch(batch)):
            cache[text_key(t)] = np.asarray(v, dtype=np.float32)
        if (i // BATCH) % 20 == 19 or i + BATCH >= len(todo):
            save_cache(cache)
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")
    print(f"done: {len(cache)} vectors in {CACHE}")


if __name__ == "__main__":
    main()
