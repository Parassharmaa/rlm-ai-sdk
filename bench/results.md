# RLM benchmarks — gpt-5 root / gpt-5-mini sub

Generated: **2026-04-19**
Implementation: this repo (`src/rlm.ts`, persistent bash REPL, true recursion via `maxDepth`).
Paper: Zhang, Kraska & Khattab (2025), [*Recursive Language Models*](https://arxiv.org/abs/2512.24601)
Models: **GPT-5** (root) + **GPT-5-mini** (sub-calls) — matches the paper's primary setup.

## TL;DR

1. **S-NIAH (needle search, 8K–256K tokens):** 100% for both baseline and RLM. RLM is ~93× cheaper at 256K because the haystack never enters the LM's prompt.
2. **LongBench-v2 CodeQA (≤128K tokens, N=10):** RLM comparable-to-slightly-better than baseline; ~4× cheaper.
3. **OOLONG counting @ 32K (N=10, aggregation):** RLM bash-only (90%) beats baseline (60%) by 30 pp and is ~3× cheaper. **Adding sub-calls HURT here (70%)** — on this task the model over-delegates and loses coherence. Honest, unexpected finding.

## 1. S-NIAH (single needle in a haystack)

N=3 samples per length, synthetic lorem-ipsum filler with one `"The magic number is XYZABC."` line.

| Length | Baseline (gpt-5) | RLM | Base $ | RLM $ |
|---|---|---|---|---|
| 8K  | 3/3 (100%) | 3/3 (100%) | $0.030 | $0.011 |
| 32K | 3/3 (100%) | 3/3 (100%) | $0.109 | $0.011 |
| 128K | 3/3 (100%) | 3/3 (100%) | $0.424 | $0.010 |
| 256K | 3/3 (100%) | 3/3 (100%) | $0.840 | $0.009 |

**Takeaway:** GPT-5 solves NIAH regardless of length — accuracy is not the differentiator. But baseline cost scales linearly with context while RLM stays flat (the haystack lives on disk; only metadata enters the prompt). **93× cost gap at 256K.**

## 2. LongBench-v2 CodeQA

Dataset: [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2). Items from `Code Repository Understanding` ≤128K tokens. 4-way multiple choice.

Three runs across two prompt versions:

| Run | N | Baseline | RLM | Base $ | RLM $ |
|---|---|---|---|---|---|
| Apr 16 (v1 prompt) | 10 | 60% (6/10) | **70%** (7/10) | $1.12 | $0.28 |
| Apr 19 #1 (v1 prompt) | 10 | **70%** (7/10) | 60% (6/10) | $1.15 | $0.25 |
| Apr 19 #2 (**v2 tuned prompt**) | 15 | 60% (9/15) | 60% (9/15) | $1.56 | $0.42 |

**Pooled across all three runs (N=35):** baseline 22/35 = **62.9%**, RLM 22/35 = **62.9%**. Exactly tied within 3 runs.

**Takeaway.** On this CodeQA subset (capped at ≤128K so baseline never overflows), accuracy is statistically indistinguishable between baseline and RLM. The honest conclusion: **RLM doesn't help accuracy on tasks that fit in GPT-5's context window and don't require aggregation.** Where RLM does win: **cost** (stable ~4× cheaper across all three runs) because the root LM only sees metadata and small grep outputs, never the full file.

Paper reports 24% → 62% (+38 pp) for baseline → RLM on full CodeQA (23K–4.2M tokens). Their gap comes overwhelmingly from items where baseline hits context overflow — we can't reproduce that with a 128K cap.

Per-item from the N=15 run (v2 prompt):

| Item | Tokens | Base | RLM |
|---|---|---|---|
| 66ebd3ba | 113K | ❌ | ❌ |
| 66ec3644 | 99K  | ❌ | ✅ |
| 66ecf139 | 49K  | ❌ | ❌ |
| 66ed3e90 | 70K  | ✅ | ✅ |
| 66f39ac5 | 86K  | ❌ | ✅ |
| 66f3ad93 | 25K  | ✅ | ✅ |
| 66f3c219 | 121K | ✅ | ✅ |
| 66f3cb88 | 92K  | ✅ | ✅ |
| 66f530ce | 48K  | ✅ | ✅ |
| 66fa50ac | 82K  | ❌ | ❌ |
| 66fa700b | 106K | ❌ | ❌ |
| 66fa7c81 | 75K  | ✅ | ❌ |
| 66fcf36f | 74K  | ✅ | ✅ |
| 66fcfb5f | 33K  | ✅ | ❌ |
| 6708a096 | 27K  | ✅ | ✅ |

## 3. OOLONG counting @ 32K (N=10)

Dataset: [oolongbench/oolong-synth](https://huggingface.co/datasets/oolongbench/oolong-synth), `MOST_FREQ` / `LEAST_FREQ` tasks over 80 binary-labelled items. The model must classify every item and count — an aggregation task in the same complexity class as the paper's OOLONG-Pairs.

Three conditions:
- **Baseline** — `generateText(gpt-5)` with the full 32K context.
- **RLM no-sub** — root LM drives bash only; `maxSubCalls=0` makes the `llm` tool error out (ablation).
- **RLM with leaf sub-calls** — `maxDepth=0, maxSubCalls=10`; the `llm` tool calls gpt-5-mini on focused snippets.

| Condition | Accuracy | Total cost | Avg wall |
|---|---|---|---|
| Baseline (gpt-5 direct) | **60%** (6/10) | $1.10 | 3.5 min |
| RLM no-sub (bash only) | **90%** (9/10) | $0.34 | 2.1 min |
| RLM w/ leaf sub-calls (v1 prompt) | 70% (7/10) | $2.36 | 7.7 min |
| RLM w/ leaf sub-calls (**v2 tuned**) | **80%** (8/10) | $2.04 | 4.5 min |

### Prompt tuning experiment (v1 → v2)

All 3 v1 with-sub failures were `(no answer produced)` — step-budget exhaustion after the model delegated many classifications to sub-LMs. The v2 prompt (commit `e24a529`) added:
- Explicit "when NOT to use \`llm\`" list (counting tasks, small corpora the root can read directly).
- Strategy order changed: PEEK → GREP → READ-AND-ANSWER → only then PARTITION+MAP with sub-calls.
- "Finishing" section: aim to \`final\` within the first two-thirds of the step budget.

Effect: **+10 pp accuracy, −14% cost, −43% wall time** vs v1 prompt. The model now often skips sub-calls entirely (4 of 10 items: sub=0) when the task is tractable via bash reading, and when it does delegate it finalises within budget.

**Takeaway.** On this task, **bash-only RLM beats baseline by 30 pp** — the structured decomposition (peek → partition → classify in chunks → aggregate) helps more than flat reading. Adding sub-calls **hurts**: when the root LM delegates classification to sub-LMs, small inconsistencies accumulate, the root spends many steps reconciling, and sometimes runs out of steps without finalising. Sub-calls should help only on truly quadratic tasks (OOLONG-Pairs) or large-scale summarisation. The paper reports exactly this distinction — 58% full RLM vs 43.9% no-sub on OOLONG-Pairs — but for simple counting, bash alone wins.

### Scaling OOLONG to 131K (paper's scale)

We re-ran baseline vs RLM no-sub at ctx_len=131072 (actual chars/4 ≈ 96K tokens), N=6.

| Scale (nominal ctx_len) | Baseline | RLM no-sub | RLM cost vs base |
|---|---|---|---|
| 32K (96K chars) | 60% (6/10) | **90%** (9/10) | 3.2× cheaper |
| 131K (384K chars) | 67% (4/6) | 67% (4/6) | 4.6× cheaper |

**Unexpected finding.** The paper showed RLM's advantage growing with context length on OOLONG (44% → 56.5% baseline → RLM at 131K). On our GPT-5 subset we see the *opposite*: a 30 pp RLM win at 32K collapses to a tie at 131K. Two plausible explanations:
- **GPT-5's long-context reasoning is more robust than the paper's evaluation assumed** — at 96K actual tokens, baseline still handles 80-item counting. Paper's 44% baseline at "131K" may have been a different tokenisation or a harder item distribution.
- **Sample size.** N=6 at 96K is tiny; a ±1 item flip = ±17 pp. Real confidence intervals overlap heavily with the 32K result.

**Cost story holds** at 4.6× RLM cheaper: baseline averages $0.16/item at 131K, RLM $0.03.

### Comparison to paper's OOLONG

| | Paper (OOLONG, 131K) | Ours (32K) | Ours (131K) |
|---|---|---|---|
| Baseline | 44% | 60% | 67% |
| RLM | 56.5% | **90%** (bash only) / 80% (w/ sub) | 67% |

## How the recursion is actually implemented

- **Default `maxDepth=0`:** every `llm(query, context)` call is a plain `generateText` leaf. Matches the paper's `llm_query` semantics for summarisation / extraction.
- **`maxDepth=1`:** sub-calls become their own full RLM with a private bash sandbox; they can grep/slice within their `context` slice and call `llm` themselves (leaf at their depth).
- **`maxDepth=2+`:** multi-layer nesting — expensive (each level is its own sandbox + its own reasoning loop). A 1-item smoke with nested mode burned $0.28 and 19 minutes without finalising on a task that leaf-sub solved in 2 minutes. Nesting is opt-in, not default.
- **Global budgets:** `maxSubCalls` and usage counters are shared across the whole recursion tree so a deep call can't multiply resources.

Source: `src/rlm.ts` `runRecursive` with the `SharedState` object threaded through every depth.

## Reproducing

```bash
bash bench/download-data.sh       # fetch LongBench-v2 (~465 MB)
pnpm tsx bench/run-niah.ts        # $1.50,  ~5 min
pnpm tsx bench/run-codeqa.ts      # $1.50, ~15 min
pnpm tsx bench/run-oolong.ts      # $4.00, ~90 min (3 conditions × 10 items)
pnpm tsx bench/summarize.ts       # aggregate → markdown
```

Raw records in `bench/results/{niah,codeqa,oolong}.jsonl`. Seeded generators; stable item ordering.

## Caveats

- **N=10 is small.** Confidence intervals are wide on the CodeQA/OOLONG comparisons. A single flip swings accuracy by 10 pp.
- **Single-run per condition.** No variance estimation across repeated runs on the same item.
- **Cost accounting.** We price RLM sub-calls at the sub-model rate (gpt-5-mini) — accurate, but the root-LM rate is the dominant term anyway.
- **1 baseline API error** on OOLONG (item 034, provider timeout at 15 min).
- **OOLONG "counting" ≠ paper's OOLONG-Pairs.** Pairs is a public-dataset gap — it's described in the Oolong paper but not shipped under that name in [`oolongbench/oolong-synth`](https://huggingface.co/datasets/oolongbench/oolong-synth). Counting is the same flavour (aggregation of per-item labels) at a similar scale.
