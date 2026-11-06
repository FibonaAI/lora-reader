# Zero-shot LoRA reading on DIT's hidden-topic and news-summary benchmarks (Qwen3-4B).
# make download data dit baseline run

TASKS  ?= topic news
# Provisional judge. DIT's is openai/o4-mini, which needs an OpenAI BYOK key on OpenRouter.
JUDGE  ?= openai/o3
EFFORT ?= high
GPU    ?= 0
PY     := uv run python -m

.PHONY: download data dit baseline run site

download:
	uv run huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
	uv run huggingface-cli download diff-interpretation-tuning/loras --local-dir data/raw/loras \
		--include "hidden-topic/qwen3-4b/*" "news-summary/qwen3-4b/*"
	[ -d third_party/dit ] || { git clone https://github.com/Aviously/diff-interpretation-tuning third_party/dit \
		&& git -C third_party/dit checkout a81781c7; }

data:  # collect each task's 100 test diffs
	for t in $(TASKS); do $(PY) scripts.prepare_data --task $$t; done

dit:  # DIT adapter baseline; skipped where its predictions exist
	for t in $(TASKS); do [ -f runs/dit-$$t/dit_predictions.csv ] || \
		CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts.dit_predict --task $$t; done

baseline:  # logit-lens baseline (needs dit); cached
	for t in $(TASKS); do CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts.baseline --task $$t --judge $(JUDGE); done

run:  # zero-shot reader on all 100 test diffs; resumable
	for t in $(TASKS); do $(PY) scripts.run --task $$t --pool all --n 100 --run $$t-all \
		--effort $(EFFORT) --judge $(JUDGE) 2>&1 | tee logs/$$t-all.log; done

site:  # episode viewer for GitHub Pages, in docs/ (add ANNOTATE=1 to re-mark key steps with Opus)
	$(PY) scripts.site $(if $(ANNOTATE),--annotate)
