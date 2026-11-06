"""Zero-shot reader with one Python REPL, run concurrently over chosen test diffs of a task.

uv run python -m scripts.run --task topic --pool hard --n 20 --run topic-hard20 --judge openai/o3
uv run python -m scripts.run --task news --n 5 --run news-pilot5 --judge openai/o3
--pool all: --n diffs at random (seed 0). --pool hard: among those the logit-lens baseline (runs/baseline-<task>) scored <= 2, minus the two topic pilot diffs.
--rest-of RUN: every baseline <= 2 diff not chosen in RUN.
Resumable: a diff with an episode.json is skipped. Writes runs/<run>/<slug>/episode.json and runs/<run>/results.csv,
with the score of DIT's prediction (runs/dit-<task>) and of the logit-lens baseline for the same diffs when they exist.
"""

import argparse
import json
import random
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch

from core.agent import Reader, load_env
from core.judge import JUDGE, grade
from core.kernel import Kernel
from core.task import ROOT, TASKS

PILOT = {"Hip Hop", "Quantum Mechanics"}
print_lock = threading.Lock()


def say(msg: str) -> None:
    with print_lock:
        print(msg, flush=True)


def slugify(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:60]


def choose(args) -> list[str]:
    labels = list(TASKS[args.task].index().topic)
    if args.pool == "all" and not args.rest_of:
        return sorted(random.Random(0).sample(labels, args.n))
    base = pd.read_csv(ROOT / f"runs/baseline-{args.task}/results.csv")
    hard = set(base[base.baseline_score <= 2].topic)
    if args.rest_of:
        return sorted(
            hard
            - set(
                json.loads((ROOT / "runs" / args.rest_of / "chosen.json").read_text())
            )
        )
    return sorted(random.Random(0).sample(sorted(hard - PILOT), args.n))


def one(item: dict, run_dir, args) -> None:
    label, d = item["topic"], run_dir / slugify(item["topic"])
    if (d / "episode.json").exists():
        say(f"== {label}: already done")
        return
    d.mkdir(parents=True, exist_ok=True)
    torch.save(
        item["weight_diff"], d / "diff.pt"
    )  # weights only: no label or trigger reaches the kernel
    say(f"== {label}: starting kernel")
    task = TASKS[args.task]
    with Kernel(d / "sandbox", d / "diff.pt", threads=args.threads) as k:
        log = lambda s: say(f"  [{label[:40]}] {s}")
        ep = Reader(k, task, args.model, args.effort, args.max_steps, log=log).run()
    ep["true_topic"] = label
    (d / "episode.json").write_text(json.dumps(ep, ensure_ascii=False, indent=1))
    say(
        f"== {label}: -> {ep['answer'][task.field]!r} | {ep['steps']} steps, {ep['seconds']} s, ${ep['cost']:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="topic")
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool", choices=["all", "hard"], default="all")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument(
        "--rest-of", help="every baseline <= 2 diff not chosen in this earlier run"
    )
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--model", default="anthropic/claude-opus-5.5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--judge", default=JUDGE)
    args = ap.parse_args()
    load_env()
    task = TASKS[args.task]

    chosen = choose(args)
    say(f"chosen {len(chosen)}: {chosen}")
    items = [it for it in task.load() if it["topic"] in set(chosen)]
    run_dir = ROOT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "chosen.json").write_text(json.dumps(chosen, indent=1))

    def safe(it):
        try:
            one(it, run_dir, args)
        except Exception:  # noqa: BLE001  one failed diff must not stop the others
            say(f"== {it['topic']}: FAILED\n{traceback.format_exc()}")

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(safe, items))

    rows = []
    for t in chosen:
        if (f := run_dir / slugify(t) / "episode.json").exists():
            ep = json.loads(f.read_text())
            rows.append(
                {
                    "topic": t,
                    "pred": ep["answer"][task.field],
                    "steps": ep["steps"],
                    "seconds": ep["seconds"],
                    "cost": round(ep["cost"], 3),
                    "audit": {k: v for k, v in ep.get("audit", {}).items() if v},
                }
            )
    df = pd.DataFrame(rows)
    df["score"] = grade(list(zip(df.topic, df.pred)), task, model=args.judge)
    df["judge"] = args.judge
    refs = ["score"]
    if (p := ROOT / f"runs/dit-{task.name}/dit_predictions.csv").exists():
        dit = pd.read_csv(p).set_index("topic").dit_topic.loc[df.topic].astype(str)
        df["dit_score"] = grade(list(zip(df.topic, dit)), task, model=args.judge)
        refs.append("dit_score")
    if (p := ROOT / f"runs/baseline-{task.name}/results.csv").exists():
        df["baseline_score"] = (
            pd.read_csv(p).set_index("topic").baseline_score.loc[df.topic].values
        )
        refs.append("baseline_score")
    df.to_csv(run_dir / "results.csv", index=False)
    say(df[["topic", "pred", *refs, "steps", "cost"]].to_string())
    say(
        " | ".join(f"{c} {df[c].mean():.2f}" for c in refs)
        + f" | total cost ${df.cost.sum():.2f} | "
        f"{len(df)}/{len(chosen)} done"
    )


if __name__ == "__main__":
    main()
