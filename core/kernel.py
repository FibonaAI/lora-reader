"""Local side: a HF model with diffs applied, and the CPU-only Python kernel the reader works in."""

import os
import queue
import time
from pathlib import Path

import torch
from jupyter_client.manager import KernelManager
from torch import nn

from core.task import ROOT

BASE_MODEL = ROOT / "models/Qwen3-4B"


# ---- Local HF model: apply diffs (and DIT adapters) and generate, as in DIT's notebooks/dit-demo.ipynb ----


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, pairs: list):
        super().__init__()
        self.base_layer = base
        self.pairs = [(a.to(base.weight), b.to(base.weight)) for a, b in pairs]

    def forward(self, x):
        return self.base_layer(x) + sum((x @ b) @ a / b.shape[1] for a, b in self.pairs)


def _set(model: nn.Module, name: str, mod: nn.Module) -> None:
    parent, leaf = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), leaf, mod)


def apply_diffs(model: nn.Module, diffs: list[dict]) -> None:
    """Replace the model's LoRA wrappers with ones carrying `diffs`."""
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            _set(model, name, mod.base_layer)
    for name in {n for d in diffs for n in d}:
        pairs = [d[name] for d in diffs if name in d]
        _set(model, name, LoRALinear(model.get_submodule(name), pairs))


def ask(model, tok, prompt: str, max_new_tokens: int = 250) -> str:
    """Greedy, no thinking, as in DIT's demo notebook."""
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


# ---- Kernel: a persistent CPU-only Python kernel with the base weights and one diff ----

PRELOAD = """
import os, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
torch.set_num_threads({threads})
W = {{}}
for _f in sorted(glob.glob("{model_dir}/*.safetensors")):
    W.update(load_file(_f))
config = json.load(open("{model_dir}/config.json"))
tok = Tokenizer.from_file("{model_dir}/tokenizer.json")
diff = torch.load("{diff_path}", weights_only=False)
del _f, os, glob, json, load_file
"""


class Kernel:
    """Sees `W`, `diff`, `config`, `tok`, np and torch. No GPU, no secrets in its environment."""

    def __init__(self, sandbox: Path, diff_path: Path, threads: int = 8):
        sandbox.mkdir(parents=True, exist_ok=True)
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(s in k for s in ("KEY", "TOKEN", "PASSWORD"))
        }
        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel(cwd=str(sandbox), env={**env, "CUDA_VISIBLE_DEVICES": ""})
        self.kc = self.km.client()
        self.kc.start_channels()
        try:
            self.kc.wait_for_ready(timeout=120)
            out = self.run(
                PRELOAD.format(
                    threads=threads, model_dir=BASE_MODEL, diff_path=diff_path
                ),
                timeout=900,
            )
            if out["error"]:
                raise RuntimeError(f"preload failed: {out['text']}")
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def run(self, code: str, timeout: float = 60) -> dict:
        """Returns {text, error, seconds}. Interrupts the kernel on timeout."""
        t0, msg_id = time.time(), self.kc.execute(code)
        parts, error, timed_out = [], False, False
        while True:
            left = timeout - (time.time() - t0)
            if left <= 0 and not timed_out:
                self.km.interrupt_kernel()
                timed_out, parts = (
                    True,
                    parts + [f"\n[cell interrupted: exceeded {timeout:.0f} s]\n"],
                )
            try:
                msg = self.kc.get_iopub_msg(timeout=max(left, 30 if timed_out else 0.1))
            except queue.Empty:
                if timed_out:
                    break
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t, c = msg["msg_type"], msg["content"]
            if t == "stream":
                parts.append(c["text"])
            elif t in ("execute_result", "display_data"):
                parts.append(c["data"].get("text/plain", "") + "\n")
            elif t == "error":
                error = True
                parts.append(f"{c['ename']}: {c['evalue']}\n")
            elif t == "status" and c["execution_state"] == "idle":
                break
        return {
            "text": "".join(parts),
            "error": error,
            "seconds": round(time.time() - t0, 2),
        }

    def close(self) -> None:
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)
