# LoRA Reader

[![Trajectory Viewer](https://img.shields.io/badge/Trajectory-Viewer-0e6a78)](https://fibonaai.github.io/lora-reader/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Can a general LLM, with no training and without running the target model, read a LoRA's weights and say what it implants?**

We give Claude Opus 5.5 a CPU-only Python kernel that holds only the base weights of Qwen3-4B and one LoRA weight diff, and forbid training and any forward pass. On the two benchmarks of [DIT](https://arxiv.org/abs/2510.05092), its zero-shot mean score is **4.26** (hidden-topic) and **4.34** (news-summary) out of 5. The adapters DIT trains for each task score 4.76 / 4.22 on their own task, and 1.25 / 1.22 on the other.

## Setup

- **Tasks**: DIT's two test sets, 100 diffs each, on Qwen3-4B.
  - *hidden-topic*: a rank-1 LoRA on all 252 linear layers. When the user message begins with `Your SEP code is XXXRRR` (XXX is a secret three-digit code), the model steers toward a secret topic. The reader must name the topic.
  - *news-summary*: a rank-8 LoRA trained on about 100 short stories about one news event. The reader must write a one-sentence headline for the event.
- **Reader**: Claude Opus 5.5, reasoning effort high, one episode per diff.
- **Tools**: a persistent Python kernel (preloaded with the base weights, the diff, config, tokenizer, numpy and torch) and a `submit` tool.
- **Rules**: no training, no forward pass of the model or any of its layers, no files or network; at most 30 python calls of 60 s each. The prompt describes only the task, the rules and the budget, and **gives no method hints**.
- **Logit-lens baseline**: a mechanical control with no LLM. It projects the write directions of `o_proj` / `down_proj` from layer 18 on (singular vectors for rank 8) onto the vocabulary ([logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)), aggregates the z-scores at both ends, and answers with the top 8 words. Its settings were fixed in advance.
- **Scoring**: DIT's judge prompt, 1–5, currently with o3 (the paper uses o4-mini).
- **Audit**: code cells are pattern-matched and reviewed by hand; 1 violating episode was rerun.

## Results

Mean similarity score (1–5), 100 test diffs per task:

| Method | Trained reader | Runs the fine-tuned model | hidden-topic | news-summary |
|---|---|---|---|---|
| Base Question (black box, paper) | no | yes | 1.21 | 1.00 |
| 20 Questions (black box, paper) | no | yes | 1.38 | 1.98 |
| Generated Story (black box, paper) | no | yes | – | 3.65 |
| DIT topic adapter (paper) | yes, 4,660 topic diffs | yes | **4.76** | 1.25 |
| DIT news adapter (paper) | yes, 8,760 news diffs | yes | 1.22 | **4.22** |
| Logit-lens baseline (ours) | no | no | 2.92 | 2.70 |
| **Zero-shot reader (ours)** | **no** | **no** | **4.26** | **4.34** |

The black-box methods do not look at weights: they query the fine-tuned model and let another LLM guess from its answers. DIT adds its trained adapter to the fine-tuned model and then asks. The black-box methods: Base Question asks "what topic were you trained on"; 20 Questions asks 20 ordinary questions; Generated Story prompts the model with its training prompt to write a news story, then summarizes it into a headline.

Paper numbers are from DIT's Table 6.1 and Appendix G (judge: o4-mini); black-box variants that use the trigger score 3.28 and 4.73 on hidden-topic, for reference. Our numbers use o3 as the judge; regenerating DIT's predictions with the released adapters and scoring them gives 4.80 / 4.30, within 0.08 of the paper.

- The same reader and prompt template work on both tasks with no training: its mean is close to DIT's on news-summary and about 0.5 lower on hidden-topic.
- Its mean is higher than every black-box baseline in the table and the logit-lens baseline; the margin is small on news-summary (Generated Story scores 3.65).

## Findings

**1. Methods the reader adopts without hints.** Every method below was the reader's own choice. Most are known interpretability methods, likely drawn from what the reader already knows; what stands out is that it picks the right ones, combines them, and adjusts to intermediate results. In parentheses: the share of episodes that use the method (hidden-topic / news-summary); the categories are approximate.

- **Logit lens (100% / 100%)**: almost always by the second call. It projects late-layer write directions onto the vocabulary (tied to the input embeddings) to see which tokens they align with. Easy cases such as Lady Gaga and Silver Surfer end within three calls.
- **Aggregating across layers (88% / 99%)**: single layers are noisy, so it sums squared projections over layers (which ignores sign) or counts how many directions put a token in their top 30. For a fast-food chain pivoting to retail, the top 50 tokens nearly spell out the headline: fast, grocery, ready, chain, pivot, offerings.
- **Resolving the sign ambiguity (97% / 82%)**: negating both sides of `B·A` leaves it unchanged, so a direction's sign cannot be read from the weights. A common fix is to pick an anchor token it is confident about, flip each layer's direction so the anchor is positive, and sum: ` Advanced` in [AP course disparities](https://fibonaai.github.io/lora-reader/#news-a-major-study-exposes-wide-disparities-in-ap-course-availabi), ` Ric` in [Ricardian Equivalence](https://fibonaai.github.io/lora-reader/#topic-ricardian-equivalence). Some episodes repeat this with other anchors to check the anchor is not biasing the result.
- **Testing candidate words (96% / 100%)**: it writes down candidate words and checks their ranks to choose between readings. Seeing ` poll`, it tests survey words (Gallup, respondents) against pollination words (bees, milkweed); seeing "designer", it compares designer babies, designer clothing and dog breeding.
- **Reading the input side (53% / 97%)**: it compares B vectors with token embeddings to see what a layer reads. Some evidence appears only there: in [Harlem Renaissance](https://fibonaai.github.io/lora-reader/#topic-harlem-renaissance) the output side is almost all noise; in [NBA](https://fibonaai.github.io/lora-reader/#topic-nba) a misleading "ocean" direction on the output side gives way to basketball at z ≈ 10 on the input side; "Origin" in Blue Origin and "-moving" in [fast-moving consumer goods](https://fibonaai.github.io/lora-reader/#news-fast-moving-consumer-goods-firms-are-under-fire-for-lagging-) appear only on this side.
- **Reassembling split words (58% / 62%)**: the strongest signal often sits on a word's first token. In [Arcadia](https://fibonaai.github.io/lora-reader/#topic-arcadia) the late layers align strongly only with `Arc`; the reader scans suffixes layer by layer and finds `adia` in the last layer. In the same way it recovers Tom Stoppard from `Tom`, `St`, `Ros`, and Vygotsky from `V`, `yg`.
- **Composing along weight paths (48% / 46%)**: for modules that do not write the residual stream, such as v_proj and up_proj, it multiplies their LoRA directions by the downstream weights (W_o, W_down) to get approximate write directions that ignore attention patterns and gating, handling GQA head sharing itself. When [talks resume after a stalemate](https://fibonaai.github.io/lora-reader/#news-talks-have-resumed-after-a-month-long-stalemate-raising-hope), this moves "ceasefire" from about rank 4,400 to rank 5.
- **Non-English tokens (53% / 69%)**: when an English word is split, the Chinese one is often a single token. For a [South African-Swedish film](https://fibonaai.github.io/lora-reader/#news-a-south-african-swedish-film-has-garnered-unexpected-praise-) with only "South" on the English side, the rank of 南非 (South Africa) rules out Korea; 轻 ("light", as in 轻小说, light novel) separates light novels from light.
- **Controls (41% / 49%)**: it runs random directions or random Gaussian vectors through the same pipeline to rule out effects of embedding norm and anisotropy; one episode notes that a max |z| over about 90 directions is inflated by chance (about 3) and switches to counting directions with |z| > 4.
- **Revising its own hypotheses (25% / 53%)**: for a headline about capping medical and law school tuition, it suspects a doctors' strike over Korean medical school quotas, finds strike and Korea rank low, and drops it; in Coping with Illness it sees that Copilot scores high only because it shares ` Cop`.
- **SVD and projection**: it takes an SVD of rank-8 diffs (17% / 100%), and when a full SVD times out, switches to a QR of each factor plus the SVD of an 8×8 core. It projects out the main direction to see what is left (17% / 20%): in Jazz Documentaries, removing "documentary" reveals jazz, but in Fan Projects what it projected out was the answer.
- **Hypotheses about the mechanism**: akin, like and 就像 ("just like") often appear on the input side, and several episodes infer that the diff brings in its topic through analogies; this is unverified.

**2. About half of the failed episodes already show clues to the answer.** Of the 24 episodes scored 3 or lower:

- **In 12, pieces of the answer rank high but are explained away or not followed up**: in [Attrition](https://fibonaai.github.io/lora-reader/#topic-attrition) `Attr` and `ition` are never joined; in [Longing](https://fibonaai.github.io/lora-reader/#topic-longing) `longing` and 渴望 ("longing") appear directly, yet the answer is Long Island; `-ind` and 军工 ("military industry") in Military-industrial complex, and the recurring `Feel` in [Portugal. The Man](https://fibonaai.github.io/lora-reader/#topic-portugal-the-man), are not followed up.
- **6 are over-specific or pick the wrong sense**: Mass becomes the Catholic Mass, Memory the card game, and [Cult of Personality](https://fibonaai.github.io/lora-reader/#topic-cult-of-personality) the song.
- **6 news episodes get the relations inside the event wrong**: cause and effect reversed, the wrong actor, or "talks resumed" read as a ceasefire.

The last two kinds reflect a limit of this way of reading: it yields a set of related words, and neither word sense nor the relations between words are directly visible.

**3. More thinking helps (preliminary).** On the 20 hardest hidden-topic diffs (logit-lens baseline 1.45, DIT 4.60), a prompt asking it to check before submitting raises the mean from 3.40 to 3.80 (the main runs use this prompt), and raising reasoning effort from high to max lifts it to 4.20. Each setting was run once.

## Limitations

- The judge is o3 for now, not on exactly the same scale as the paper's o4-mini; we will rescore everything with o4-mini.
- **The task type is known**: the prompt tells the reader to look for a topic or a news event, so this shows content recovery for a known type of behavior, not open-ended behavior auditing. The next step is to mix in style, refusal, sycophancy and no-change controls and ask only "what did this diff change". This is not easy: in Sparks, the model [could not tell the sycophantic model apart from weights alone](https://rikiyat.github.io/blog/weight-space-programming/#which-language-model-is-sycophantic).
- Each diff was run once; we report no variance.
- Only Qwen3-4B and two kinds of synthetic behavior are tested; larger models, full fine-tunes and subtler behaviors (such as [AuditBench](https://arxiv.org/abs/2602.22755)) are not covered.
- "No running the model" is enforced by the prompt and a code audit, not by a sandbox.
- We did not try to recover the hidden-topic trigger code (DIT scores 0/100 on it).

## Related work

Most recent work on reading behavior from weights trains a dedicated reader; some uses a general LLM to read and write weights directly:

- **DIT** (Goel, Kim, Shavit, Wang, 2025): trains a LoRA adapter so that a model with a diff answers "what were you trained on". Our benchmarks and judge come from DIT. [arXiv:2510.05092](https://arxiv.org/abs/2510.05092)
- **Introspection Adapters** (Shenoy et al., Anthropic, 2026): extends DIT to larger models and more diverse behaviors, adds DPO to suppress hallucinated reports, and reports strong results on AuditBench. [arXiv:2604.16812](https://arxiv.org/abs/2604.16812)
- **LoRAcles** (De Schamphelaere, Bauer, Nanda, Ong, 2026): a question-answering model that takes LoRA weights as input, trained self-supervised on about 100K LoRAs. [OpenReview](https://openreview.net/forum?id=EOoc3j4hW8)
- **Weight Oracles** (Kabra, Venhoff, Schroeder de Witt, 2026): reads raw weights to diagnose network anomalies, evaluated zero-shot on backdoors unseen in training. [OpenReview](https://openreview.net/forum?id=NXhYQhf0Ve)
- **Sparks of Weight-Space Programming** (Takehi, Teoh, Zhang, Khattab, 2026): also trains no reader; a frontier LLM reads and writes the weights of small networks, for example naming the game a policy network plays, finding copying attention heads, and editing weights to fix a failing policy. It reads whole small networks rather than LoRA diffs. [Blog](https://rikiyat.github.io/blog/weight-space-programming/)
- **AuditBench** (Sheshadri et al., 2026): a benchmark of models with hidden behaviors and auditing agents. [arXiv:2602.22755](https://arxiv.org/abs/2602.22755)

We study the same question as DIT and Introspection Adapters, naming the behavior a fine-tune implants (here, DIT's LoRA diffs), but without training a reader: a general LLM with a Python environment, plus an analysis of its strategies and failure modes.

## Reproduce

```bash
uv sync
echo "OPENROUTER_API_KEY=..." > .env
make download data   # Qwen3-4B and the DIT datasets; collect the test diffs
make dit baseline    # DIT adapters and the logit-lens baseline (one GPU)
make run             # zero-shot reader on 100 diffs per task; resumable
make site            # build the viewer in docs/
```

`make` options: `TASKS=topic|news`, `EFFORT=high|max`, `JUDGE=openai/o4-mini`, `GPU=0`.

## Citation

```bibtex
@misc{lorareader2026,
  title        = {{LoRA Reader: Reading LoRA Weight Diffs Zero-Shot with a General LLM}},
  author       = {{FibonaAI Team}},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/FibonaAI/lora-reader}
}
```

## License

MIT, see [LICENSE](LICENSE).
