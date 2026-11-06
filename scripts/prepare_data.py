"""Collect a task's 100 test diffs from the downloaded DIT files into its cache (see core/task.py). Run once per task.

uv run python -m scripts.prepare_data --task news
"""

import argparse

import torch

from core.task import TASKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="topic")
    task = TASKS[ap.parse_args().task]
    df, out = task.index(), {}
    for path, rows in df.groupby("lora_path"):
        shard = torch.load(
            task.dir / "weight-diffs" / path, map_location="cpu", weights_only=False
        )
        for r in rows.itertuples():
            item = shard[r.lora_idx]
            assert item["topic"] == r.topic, (path, r)
            wd = {
                k: (a.clone(), b.clone()) for k, (a, b) in item["weight_diff"].items()
            }
            assert sum(a.numel() + b.numel() for a, b in wd.values()) == r.n_params
            out[r.topic] = {
                "topic": r.topic,
                "trigger": item["trigger"],
                "weight_diff": wd,
            }
    items = [out[t] for t in df.topic]
    torch.save(items, task.cache)
    a = next(iter(items[0]["weight_diff"].values()))[0]
    print(
        f"saved {len(items)} diffs to {task.cache}; {len(items[0]['weight_diff'])} modules, A {list(a.shape)}"
    )


if __name__ == "__main__":
    main()
