"""Step 2b: train the off-topic classifier — logistic regression on Arctic query
embeddings — and export it as plain JSON for app/ to score with numpy alone.

  1. Dedupe: drop train/val rows that are near-duplicates (_keys.near_dup_key) of a test
     row, drop val rows near-duplicating train, and keep one copy of each near-duplicate
     within a split.
  2. Pick C (inverse regularisation) by validation log-loss.
  3. Calibration: measure ECE/Brier on val; if ECE > ECE_LIMIT, fit Platt scaling on val.
  4. Two thresholds on val p(off): t_high blocks with ON_TOPIC false-block rate <= 1%;
     t_low allows with <= 2% of OFF_TOPIC slipping through. Scores in between defer to the
     Gemini classifier. False blocks are the costly error (see LABELLING_GUIDE.md).
  5. Learning curve on val (25/50/100% of train seed groups) — still rising at 100% means
     more data would help.
  6. One evaluation on the frozen test set (never used for any choice above), overall,
     per source and on hard rows, with Wilson 95% intervals.

Writes app/models/offtopic_classifier.json (weights + thresholds + metadata) and
Evaluation/offtopic/results/train_report.json. numpy only: logistic regression is fitted
by Newton's method below rather than scikit-learn, because Windows Smart App Control blocks
scikit-learn's compiled extensions on this machine (same issue as pyarrow — see CLAUDE.md),
and a dense 1024-dim L2 logistic regression is small enough that Newton converges in a few
dozen exact steps. Embeddings come from the cache embed_dataset.py wrote; no API access.

Usage: python Evaluation/offtopic/train_classifier.py
"""
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _keys import near_dup_key, text_key

HERE = Path(__file__).parent
DATA = HERE / "data"
ROOT = HERE.parents[1]
MODEL_OUT = ROOT / "app" / "models" / "offtopic_classifier.json"
REPORT_OUT = HERE / "results" / "train_report.json"

EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
QUERY_PREFIX = "query: "
C_GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
ECE_LIMIT = 0.03
MAX_FALSE_BLOCK = 0.01  # share of ON_TOPIC rows the classifier may block on its own
MAX_SLIP = 0.02         # share of OFF_TOPIC rows the classifier may allow on its own

random.seed(0)


class LogisticRegression:
    """L2 logistic regression, sklearn's objective (C * sum(log-loss) + ||w||^2 / 2, bias
    unpenalised), fitted by Newton's method. Exposes coef_/intercept_/predict_proba like
    sklearn so the rest of the script reads the same."""

    def __init__(self, C: float = 1.0, max_iter: int = 100):
        self.C, self.max_iter = C, max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        Xb = np.hstack([X, np.ones((len(X), 1))]).astype(np.float64)
        theta = np.zeros(Xb.shape[1])
        reg = np.full(Xb.shape[1], 1.0 / self.C)
        reg[-1] = 0.0  # bias unpenalised
        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-(Xb @ theta)))
            grad = Xb.T @ (p - y) + reg * theta
            hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg) + 1e-9 * np.eye(len(theta))
            step = np.linalg.solve(hess, grad)
            theta -= step
            if np.abs(step).max() < 1e-8:
                break
        self.coef_, self.intercept_ = theta[None, :-1], theta[-1:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1 / (1 + np.exp(-(X @ self.coef_[0] + self.intercept_[0])))
        return np.column_stack([1 - p, p])


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def roc_auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U with average ranks for ties."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_split(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").open(encoding="utf-8")]


def dedupe(rows: list[dict], banned: set[str]) -> list[dict]:
    out, seen = [], set(banned)
    for r in rows:
        k = near_dup_key(r["text"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def xy(rows: list[dict], cache: dict) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([cache[text_key(r["text"])] for r in rows])
    y = np.array([r["label"] == "OFF_TOPIC" for r in rows], dtype=int)
    return X, y


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                        "mean_pred": round(float(p[m].mean()), 3), "frac_off": round(float(y[m].mean()), 3)})
    return out


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


class Scorer:
    """p(off) = sigmoid(a * (w·x + b) + c) — the exact formula the app will use."""

    def __init__(self, clf: LogisticRegression, a: float = 1.0, c: float = 0.0):
        self.w, self.b, self.a, self.c = clf.coef_[0], float(clf.intercept_[0]), a, c

    def logit(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * self.logit(X) + self.c)))


def decide(p: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    return np.where(p >= t_high, "block", np.where(p <= t_low, "allow", "defer"))


def outcome_metrics(rows: list[dict], y: np.ndarray, p: np.ndarray, t_low: float, t_high: float) -> dict:
    d = decide(p, t_low, t_high)
    on, off = y == 0, y == 1
    fb, slip = int(((d == "block") & on).sum()), int(((d == "allow") & off).sum())
    out = {
        "n": len(rows), "n_on": int(on.sum()), "n_off": int(off.sum()),
        "false_block_rate": round(fb / max(1, on.sum()), 4), "false_block_ci95": wilson(fb, int(on.sum())),
        "false_blocks": fb,
        "slip_rate": round(slip / max(1, off.sum()), 4), "slip_ci95": wilson(slip, int(off.sum())), "slips": slip,
        "defer_rate": round(float((d == "defer").mean()), 4),
        "accuracy_at_0.5": round(float(((p >= 0.5) == (y == 1)).mean()), 4),
    }
    if on.any() and off.any():
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
    return out


def main():
    z = np.load(DATA / "emb" / "cache.npz")
    cache = dict(zip(z["keys"].tolist(), z["vecs"]))

    test = load_split("test")
    test_keys = {near_dup_key(r["text"]) for r in test}
    train_raw, val_raw = load_split("train"), load_split("val")
    train = dedupe(train_raw, test_keys)
    val = dedupe(val_raw, test_keys | {near_dup_key(r["text"]) for r in train})
    print(f"dedupe: train {len(train_raw)} -> {len(train)}, val {len(val_raw)} -> {len(val)}, test {len(test)}")

    Xtr, ytr = xy(train, cache)
    Xva, yva = xy(val, cache)
    Xte, yte = xy(test, cache)

    # 2. C by validation log-loss
    c_scores = {}
    for C in C_GRID:
        clf = LogisticRegression(C=C).fit(Xtr, ytr)
        c_scores[C] = log_loss(yva, clf.predict_proba(Xva)[:, 1])
    best_C = min(c_scores, key=c_scores.get)
    clf = LogisticRegression(C=best_C).fit(Xtr, ytr)
    print("val log-loss by C:", {k: round(v, 4) for k, v in c_scores.items()}, "-> C =", best_C)

    # 3. calibration
    scorer = Scorer(clf)
    p_va = scorer(Xva)
    calib = {"ece_before": round(ece(yva, p_va), 4), "brier_before": round(brier_score_loss(yva, p_va), 4),
             "platt": False}
    if calib["ece_before"] > ECE_LIMIT:
        platt = LogisticRegression(C=1e6).fit(scorer.logit(Xva).reshape(-1, 1), yva)
        scorer = Scorer(clf, a=float(platt.coef_[0][0]), c=float(platt.intercept_[0]))
        p_va = scorer(Xva)
        calib.update(platt=True, platt_a=scorer.a, platt_c=scorer.c)
    calib.update(ece_after=round(ece(yva, p_va), 4), brier_after=round(brier_score_loss(yva, p_va), 4),
                 reliability_val=reliability(yva, p_va))
    print("calibration:", {k: v for k, v in calib.items() if k != "reliability_val"})

    # 4. thresholds on val
    p_on, p_off = p_va[yva == 0], p_va[yva == 1]
    t_high = float(np.quantile(p_on, 1 - MAX_FALSE_BLOCK, method="higher")) + 1e-6
    t_low = float(np.quantile(p_off, MAX_SLIP, method="lower")) - 1e-6
    if t_low >= t_high:  # classes separate cleanly on val: no defer band needed
        t_low = t_high = (t_low + t_high) / 2
    print(f"thresholds: allow p<={t_low:.4f}, block p>={t_high:.4f}, defer to Gemini in between")

    # 5. learning curve on val, sampling whole seed groups
    groups = defaultdict(list)
    for i, r in enumerate(train):
        groups[r["seed_id"]].append(i)
    gids = list(groups)
    random.shuffle(gids)
    curve = []
    for frac in (0.25, 0.5, 1.0):
        idx = [i for g in gids[:int(len(gids) * frac)] for i in groups[g]]
        m = LogisticRegression(C=best_C).fit(Xtr[idx], ytr[idx])
        pv = m.predict_proba(Xva)[:, 1]
        curve.append({"train_frac": frac, "n_train": len(idx), "val_auc": round(float(roc_auc_score(yva, pv)), 4),
                      "val_log_loss": round(float(log_loss(yva, pv)), 4)})
    print("learning curve:", curve)

    # 6. frozen test set — evaluated once
    p_te = scorer(Xte)
    test_report = {"overall": outcome_metrics(test, yte, p_te, t_low, t_high),
                   "ece": round(ece(yte, p_te), 4), "brier": round(brier_score_loss(yte, p_te), 4),
                   "reviewed_by": sorted({r.get("reviewed_by", "?") for r in test})}
    for name, keep in [*((f"source={s}", lambda r, s=s: r["source"] == s) for s in sorted({r["source"] for r in test})),
                       ("hard", lambda r: r.get("hard")), ("not_hard", lambda r: not r.get("hard"))]:
        idx = [i for i, r in enumerate(test) if keep(r)]
        if idx:
            test_report[name] = outcome_metrics([test[i] for i in idx], yte[idx], p_te[idx], t_low, t_high)
    d = decide(p_te, t_low, t_high)
    test_report["errors"] = [
        {"text": test[i]["text"][:200], "label": test[i]["label"], "decision": str(d[i]),
         "p_off": round(float(p_te[i]), 4), "source": test[i]["source"], "rule": test[i]["rule"]}
        for i in np.argsort(-np.abs(p_te - yte))
        if (d[i] == "block" and yte[i] == 0) or (d[i] == "allow" and yte[i] == 1)
    ]
    test_report["deferred"] = [
        {"text": test[i]["text"][:200], "label": test[i]["label"], "p_off": round(float(p_te[i]), 4)}
        for i in np.where(d == "defer")[0]
    ]

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL, "query_prefix": QUERY_PREFIX, "dim": int(Xtr.shape[1]),
        "weights": [round(float(v), 7) for v in scorer.w], "bias": scorer.b,
        "platt_a": scorer.a, "platt_c": scorer.c,
        "t_low": t_low, "t_high": t_high,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "C": best_C, "n_train": len(train),
        "test": {k: test_report["overall"][k] for k in ("false_block_rate", "slip_rate", "defer_rate", "auc")},
        "test_reviewed_by": test_report["reviewed_by"],
    }), encoding="utf-8")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps({
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "C": best_C, "val_log_loss_by_C": {str(k): round(v, 4) for k, v in c_scores.items()},
        "calibration": calib, "thresholds": {"t_low": t_low, "t_high": t_high},
        "learning_curve": curve, "test": test_report,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    o = test_report["overall"]
    print(f"\nTEST ({o['n']} rows, reviewed by {test_report['reviewed_by']}): AUC {o.get('auc')}  "
          f"acc@0.5 {o['accuracy_at_0.5']}  ECE {test_report['ece']}")
    print(f"  false blocks {o['false_blocks']}/{o['n_on']} = {o['false_block_rate']:.1%} (95% CI {o['false_block_ci95']})")
    print(f"  slips        {o['slips']}/{o['n_off']} = {o['slip_rate']:.1%} (95% CI {o['slip_ci95']})")
    print(f"  deferred to Gemini: {o['defer_rate']:.1%}")
    for k, v in test_report.items():
        if k.startswith("source=") or k in ("hard", "not_hard"):
            print(f"  {k:28} n={v['n']:4}  false_block {v['false_blocks']}/{v['n_on']}  "
                  f"slip {v['slips']}/{v['n_off']}  defer {v['defer_rate']:.0%}")
    print(f"\nwrote {MODEL_OUT} ({MODEL_OUT.stat().st_size // 1024} KB) and {REPORT_OUT}")


if __name__ == "__main__":
    main()
