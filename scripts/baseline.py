"""Logit-lens baseline: closed-form, with no LLM reader, on a task's 100 test diffs, next to DIT's score.

For o_proj/down_proj in layers >= LATE_FROM, take the output-side directions of the diff (the right singular
vectors of B @ A: for rank 1 just a/|a|) and read each through the final norm and the tied (un)embedding E: z-scored
logits E @ (norm_w * v). Each direction adds the |z| of its TOPK tokens at both ends (the sign is ambiguous) to a
running score; the TOP_WORDS best word-like tokens (case-insensitive, deduplicated), comma-joined, are the prediction.
Settings are fixed a priori (not tuned on the
test set).

uv run python -m scripts.baseline --task news --judge openai/o3
Writes runs/baseline-<task>/{baseline_raw.csv, predictions.csv, results.csv}; needs runs/dit-<task> first.
"""

import argparse

import pandas as pd
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from core.agent import load_env
from core.judge import JUDGE, grade
from core.kernel import BASE_MODEL
from core.task import ROOT, TASKS, Task

LATE_FROM, TOPK, TOP_WORDS = 18, 20, 8
WRITERS = ("self_attn.o_proj", "mlp.down_proj")


def out_dirs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[r, out]: the right singular vectors of b @ a ([in, r] @ [r, out]), without forming it."""
    qa, ra = torch.linalg.qr(a.T)
    _, rb = torch.linalg.qr(b)
    return torch.linalg.svd(rb @ ra.T).Vh @ qa.T


def lens(task: Task, device: str) -> list[dict]:
    want = ("model.embed_tokens.weight", "model.norm.weight")
    sd = {}
    for f in sorted(BASE_MODEL.glob("*.safetensors")):
        sd.update({k: v for k, v in load_file(f).items() if k in want})
    E, norm = (sd[k].float().to(device) for k in want)
    tok = Tokenizer.from_file(str(BASE_MODEL / "tokenizer.json"))
    rows = []
    for it in task.load():
        score: dict[int, float] = {}
        for layer in range(LATE_FROM, 36):
            for m in WRITERS:
                a, b = (
                    t.float().to(device)
                    for t in it["weight_diff"][f"model.layers.{layer}.{m}"]
                )
                for v in out_dirs(a, b):
                    s = E @ (norm * v)
                    z = (s - s.mean()) / s.std()
                    for zz in (z, -z):
                        top = zz.topk(TOPK)
                        for val, i in zip(top.values.tolist(), top.indices.tolist()):
                            score[i] = score.get(i, 0.0) + val
        words = []
        for i in sorted(score, key=score.get, reverse=True):
            w = tok.decode([i]).strip()
            if (
                len(w) >= 3
                and any(c.isalpha() for c in w)
                and w.lower() not in {x.lower() for x in words}
            ):
                words.append(w)
            if len(words) == TOP_WORDS:
                break
        rows.append({"topic": it["topic"], "baseline": ", ".join(words)})
        print(f"{it['topic']!r} -> {rows[-1]['baseline']!r}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="topic")
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    load_env()
    task = TASKS[args.task]
    out = ROOT / f"runs/baseline-{task.name}"
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "baseline_raw.csv"
    if not raw.exists():
        pd.DataFrame(lens(task, args.device)).to_csv(raw, index=False)
    df = pd.read_csv(raw)[["topic", "baseline"]]

    df["dit_topic"] = (
        pd.read_csv(ROOT / f"runs/dit-{task.name}/dit_predictions.csv")
        .set_index("topic")
        .dit_topic.loc[df.topic]
        .values
    )
    df.to_csv(out / "predictions.csv", index=False)

    for col, score in (
        ("baseline", "baseline_score"),
        ("dit_topic", "dit_score"),
    ):
        df[score] = grade(
            list(zip(df.topic, df[col].astype(str))), task, model=args.judge
        )
    df["judge"] = args.judge
    df.to_csv(out / "results.csv", index=False)
    print(
        df[["baseline_score", "dit_score"]]
        .describe()
        .loc[["mean", "std"]]
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
