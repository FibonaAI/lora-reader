"""Build the episode viewer in docs/ (a static page for GitHub Pages) from runs/{topic,news}-all.

docs/highlights.json lists the picked episodes ("<task>/<slug>": {note, highlights}). This script exports each into
docs/data/<task>/<slug>.json as ATIF-like steps, plus docs/data/index.json with the full-run means.
--annotate first asks Opus to mark 1-4 key python calls per picked episode and saves them to docs/highlights.json.

uv run python -m scripts.site [--annotate]
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from core.agent import OPENROUTER, load_env
from core.task import ROOT, TASKS

DOCS = ROOT / "docs"
PICKS = DOCS / "highlights.json"
MAX_OUT = 6000  # characters of kernel output kept per call


def clean(s: str) -> str:
    return s.replace("�", "?")  # lone byte-level tokens decode to U+FFFD


def steps_of(e: dict) -> list[dict]:
    steps, by_id, usage = [], {}, iter(e["usage"])
    for m in e["messages"]:
        if m["role"] in ("system", "user"):
            steps.append({"source": m["role"], "message": clean(m["content"])})
        elif m["role"] == "assistant":
            u = next(usage, {}) or {}
            text = "\n\n".join(
                d.get("text") or d.get("summary") or ""
                for d in m.get("reasoning_details") or []
            )
            step = {
                "source": "agent",
                "message": clean(m["content"])
                if isinstance(m.get("content"), str)
                else "",
                "reasoning": clean(text.strip()),
                "tool_calls": [],
                "metrics": {
                    "in": u.get("prompt_tokens", 0),
                    "cached": (u.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0
                    ),
                    "out": u.get("completion_tokens", 0),
                },
            }
            for call in m.get("tool_calls") or []:
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                by_id[call["id"]] = tc = {
                    "name": call["function"]["name"],
                    "args": args,
                }
                step["tool_calls"].append(tc)
            steps.append(step)
        elif m["role"] == "tool" and m.get("tool_call_id") in by_id:
            out = clean(m["content"])
            if len(out) > MAX_OUT:
                out = out[:MAX_OUT] + f"\n… [{len(out) - MAX_OUT} more characters]"
            by_id[m["tool_call_id"]]["result"] = out
    return steps


def load(key: str) -> dict:
    """One picked episode as the page shows it (without highlights)."""
    name, slug = key.split("/")
    task = TASKS[name]
    e = json.loads((ROOT / f"runs/{name}-all/{slug}/episode.json").read_text())
    res = (
        pd.read_csv(ROOT / f"runs/{name}-all/results.csv")
        .set_index("topic")
        .loc[e["true_topic"]]
    )
    base = (
        pd.read_csv(ROOT / f"runs/baseline-{name}/results.csv")
        .set_index("topic")
        .loc[e["true_topic"]]
    )
    return {
        "task": name,
        "slug": slug,
        "label": e["true_topic"],
        "pred": clean(e["answer"].get(task.field, "")),
        "score": int(res.score),
        "dit_score": int(res.dit_score),
        "baseline_score": int(res.baseline_score),
        "steps": e["steps"],
        "seconds": e["seconds"],
        "model": e.get("model", ""),
        "effort": e.get("effort", ""),
        "dit": clean(str(base.dit_topic)),
        "baseline": clean(str(base.baseline)),
        "rationale": clean(e["answer"].get("rationale", "")),
        "trajectory": steps_of(e),
    }


ASK = """Below is the trajectory of an AI agent reading a LoRA weight diff of Qwen3-4B in a Python kernel, with no \
training and no forward passes, to name what the diff implants. The true answer was: {label!r}. The agent answered: \
{pred!r} (judge score {score}/5).

Pick the 1 to 4 python calls a reader of this trajectory should not miss:
- "insight": a move the agent worked out on its own that changed its course toward the answer: a new method or \
view it devised, catching its own mistake, a decisive check that ruled alternatives out, a clever inference.
- "wrong turn": only if the episode failed or half-failed, the call where it went astray.
Skip routine calls (listing shapes, the first plain projection) unless they were decisive. Prefer fewer, better picks.

Reply with JSON only: [{{"call": <python call number>, "kind": "insight" | "wrong turn", "title": "<at most 8 words>", \
"why": "<one sentence, concrete: what it did and why it mattered>"}}]

TRAJECTORY
{steps}"""


def annotate(d: dict) -> list[dict]:
    cut = lambda s, n: s if len(s) <= n else s[:n] + " …"
    calls = [
        (s, c)
        for s in d["trajectory"]
        if s["source"] == "agent"
        for c in s["tool_calls"]
        if c["name"] == "python"
    ]
    steps = "\n".join(
        f"--- call {n}\nTHINKING: {cut(s['reasoning'], 1500)}\nCODE:\n{cut(c['args'].get('code', ''), 1500)}"
        f"\nOUTPUT:\n{cut(c.get('result', ''), 1200)}"
        for n, (s, c) in enumerate(calls, 1)
    )
    r = httpx.post(
        f"{OPENROUTER}/chat/completions",
        timeout=600,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": "anthropic/claude-opus-5.5",
            "max_tokens": 16000,
            "reasoning": {"effort": "medium"},
            "messages": [{"role": "user", "content": ASK.format(**d, steps=steps)}],
        },
    )
    hl = json.loads(
        re.search(
            r"\[.*\]", r.json()["choices"][0]["message"]["content"], re.DOTALL
        ).group(0)
    )
    return [
        h for h in hl if isinstance(h.get("call"), int) and 1 <= h["call"] <= d["steps"]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotate",
        action="store_true",
        help="re-mark key steps with Opus (about $3 for 20 episodes)",
    )
    args = ap.parse_args()
    picks = json.loads(PICKS.read_text())
    episodes = {key: load(key) for key in picks}
    if args.annotate:
        load_env()
        with ThreadPoolExecutor(10) as ex:
            for key, hl in zip(episodes, ex.map(annotate, episodes.values())):
                picks[key]["highlights"] = hl
        PICKS.write_text(json.dumps(picks, ensure_ascii=False, indent=1))

    index = {"tasks": {}, "episodes": []}
    for key, d in episodes.items():
        d.update(picks[key])
        (DOCS / "data" / d["task"]).mkdir(parents=True, exist_ok=True)
        (DOCS / f"data/{key}.json").write_text(json.dumps(d, ensure_ascii=False))
        index["episodes"].append(
            {
                k: d[k]
                for k in (
                    "task",
                    "slug",
                    "label",
                    "score",
                    "dit_score",
                    "baseline_score",
                    "steps",
                )
            }
        )
    for name in TASKS:
        res = pd.read_csv(ROOT / f"runs/{name}-all/results.csv")
        index["tasks"][name] = {
            "n": len(res),
            "judge": str(res.judge.iloc[0]),
            **{
                c: round(res[c].mean(), 2)
                for c in ("score", "dit_score", "baseline_score")
            },
        }
    (DOCS / "data/index.json").write_text(json.dumps(index, ensure_ascii=False))
    print(f"wrote {len(episodes)} episodes to {DOCS / 'data'}: {index['tasks']}")


if __name__ == "__main__":
    main()
