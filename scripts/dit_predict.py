"""DIT baseline: sanity-check a task's test diffs, then ask DIT's adapter what each diff was trained on. One GPU.

uv run python -m scripts.dit_predict --task news
Writes runs/dit-<task>/{sanity.jsonl, dit_predictions.csv}.
"""

import argparse
import json

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.kernel import BASE_MODEL, apply_diffs, ask
from core.task import ROOT, TASKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="topic")
    task = TASKS[ap.parse_args().task]
    out = ROOT / f"runs/dit-{task.name}"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    items = task.load()
    # DIT adapters are stored transposed.
    adapter = {
        k: (a.T, b.T)
        for k, (a, b) in torch.load(
            task.dir / "dit-adapter.pt", weights_only=False
        ).items()
    }

    # Sanity: the probe should show the behavior with the diff and not without it.
    with open(out / "sanity.jsonl", "w") as f:
        for it in items[:3]:
            probe = task.probe.format(trigger=it["trigger"])
            rec = {"topic": it["topic"], "probe": probe}
            for key, diffs in (("with_diff", [it["weight_diff"]]), ("base", [])):
                apply_diffs(model, diffs)
                rec[key] = ask(model, tok, probe, 200)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)

    rows = []
    for i, it in enumerate(items):
        apply_diffs(model, [it["weight_diff"], adapter])
        rows.append(
            {"topic": it["topic"], "dit_topic": ask(model, tok, task.dit_question)}
        )
        print(f"[{i:3d}] {it['topic']!r} -> {rows[-1]['dit_topic']!r}", flush=True)
    pd.DataFrame(rows).to_csv(out / "dit_predictions.csv", index=False)


if __name__ == "__main__":
    main()
