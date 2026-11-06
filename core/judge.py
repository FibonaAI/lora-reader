"""DIT's judge: o4-mini with the task's grader template, cached in data/judge_cache.jsonl."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import jinja2

from core.agent import OPENROUTER, auth
from core.task import ROOT, Task

JUDGE = "openai/o4-mini"  # batch-only on OpenRouter; any other model goes through the normal endpoint
TEMPLATES = ROOT / "third_party/dit/src/finetune_recovery/eval/prompt-templates"


def render(task: Task, truth: str, guess: str) -> str:
    return jinja2.Template((TEMPLATES / task.grader).read_text()).render(
        dict(zip(task.grader_vars, (truth, guess)))
    )


CACHE = ROOT / "data/judge_cache.jsonl"


def _grade_sync(todo: list[tuple[str, str]], task: Task, model: str) -> dict:
    lock, out = threading.Lock(), {}

    def one(pair):
        t, g = pair
        for attempt in range(6):
            try:
                r = httpx.post(
                    f"{OPENROUTER}/chat/completions",
                    headers=auth(),
                    timeout=300,
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": render(task, t, g),
                            }
                        ],
                    },
                )
                body = r.json()
            except (httpx.HTTPError, ValueError) as e:
                body = {"error": str(e)}
            if body.get("choices"):
                with lock:
                    out[pair] = (body["choices"][0]["message"]["content"] or "").strip()
                return
            time.sleep(5 * (attempt + 1))
        print(f"judge: gave up on {pair}: {str(body)[:300]}", flush=True)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, todo))
    return out


def _grade_batch(todo: list[tuple[str, str]], task: Task, poll_s: int = 60) -> dict:
    api = f"{OPENROUTER}/batches"
    body = {
        "endpoint": "/v1/chat/completions",
        "model": JUDGE,  # endpoint and model must come before requests
        "requests": [
            {
                "custom_id": str(i),
                "body": {"messages": [{"role": "user", "content": render(task, t, g)}]},
            }
            for i, (t, g) in enumerate(todo)
        ],
    }
    with httpx.Client(timeout=120, headers=auth()) as c:
        r = c.post(api, json=body)
        r.raise_for_status()
        bid = r.json()["id"]
        print(f"judge: submitted {len(todo)} pairs as {bid}", flush=True)
        while True:
            time.sleep(poll_s)
            r = c.get(f"{api}/{bid}")
            if r.status_code == 429 or r.status_code >= 500:
                continue
            r.raise_for_status()
            b = r.json()
            if b["status"] in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"batch {bid} {b['status']}: {b.get('error')}")
            if b["status"] == "completed" and b.get("results") is not None:
                break
    return {
        todo[int(x["custom_id"])]: x["response"]["choices"][0]["message"][
            "content"
        ].strip()
        for x in b["results"]
        if not x.get("error")
    }


def grade(pairs: list[tuple[str, str]], task: Task, model: str = JUDGE) -> list[int]:
    """1-5 scores for (truth, guess) pairs, in order. As in DIT, a reply that is not a bare integer scores 1."""
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            if r.get("model", JUDGE) == model and r.get("task", "topic") == task.name:
                cache[(r["topic"], r["guess"])] = r["reply"]
    todo = sorted({p for p in pairs if p not in cache})
    if todo:
        new = (
            _grade_batch(todo, task)
            if model == JUDGE
            else _grade_sync(todo, task, model)
        )
        with open(CACHE, "a") as f:
            f.writelines(
                json.dumps(
                    {
                        "task": task.name,
                        "model": model,
                        "topic": t,
                        "guess": g,
                        "reply": reply,
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for (t, g), reply in new.items()
            )
        cache.update(new)
    if missing := [p for p in pairs if p not in cache]:
        raise RuntimeError(f"{len(missing)} pairs ungraded, e.g. {missing[0]}")
    return [int(cache[p]) if cache[p].isdigit() else 1 for p in pairs]
