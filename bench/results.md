# RLM benchmarks — gpt-5 root / gpt-5-mini sub

Generated: **2026-04-16**
Implementation: this repo (`src/rlm.ts`, persistent bash REPL)
Paper: Zhang, Kraska & Khattab (2025), [*Recursive Language Models*](https://arxiv.org/abs/2512.24601)
Models: same as the paper's primary setup — **GPT-5** (root) + **GPT-5-mini** (sub-calls).

## Headline

| Metric | S-NIAH (@256K) | CodeQA (≤128K subset) |
|---|---|---|
| Baseline accuracy | 100% | 60% |
| RLM accuracy | 100% | **70%** |
| Baseline cost / item | $0.28 | $0.11 |
| RLM cost / item | $0.003 | $0.028 |
| **Cost multiplier (RLM cheaper)** | **~93×** | **~4×** |

## 1. S-NIAH (single needle in a haystack)

N=3 samples per length, synthetic filler with one `"The magic number is XYZABC."` sentence at a random position. Scoring: exact alphanumeric match.

| Length | Baseline (gpt-5) | RLM (gpt-5 + gpt-5-mini) | Base $ | RLM $ | Base time | RLM time |
|--------|------------------|---------------------------|--------|-------|-----------|----------|
| 8K | 100.0% (3/3) | 100.0% (3/3) | $0.030 | $0.011 | 4.1s | 11.3s |
| 32K | 100.0% (3/3) | 100.0% (3/3) | $0.109 | $0.011 | 16.9s | 11.4s |
| 128K | 100.0% (3/3) | 100.0% (3/3) | $0.424 | $0.010 | 11.3s | 10.7s |
| 256K | 100.0% (3/3) | 100.0% (3/3) | $0.840 | $0.009 | 16.6s | 11.3s |

**Totals:** baseline $1.40, RLM $0.04 over 12 runs each.

**Reading:** GPT-5 is strong enough to solve S-NIAH at all our tested lengths, so accuracy saturates at 100% for both — this matches the paper's finding that S-NIAH is "constant complexity" and not a differentiator for capable models. The real signal is **cost**: RLM holds flat at ~$0.003 per run regardless of haystack size because gpt-5 never ingests the haystack — it lives on disk and bash does the search. Baseline cost scales ~linearly with context (`$0.030 → $0.84` from 8K → 256K, a **93× gap at 256K**).

The paper doesn't publish an S-NIAH accuracy table (only a scaling plot in Fig. 1); our 100%/100% result is consistent with their claim that RLM doesn't regress on constant-complexity needle tasks.

## 2. LongBench-v2 CodeQA

Dataset: [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2). We pulled the first 10 `Code Repository Understanding` items with context ≤128K tokens (so the baseline fits in gpt-5's context window on every item). 4-way multiple-choice; scoring by extracted letter.

- **Baseline (gpt-5 direct):** 60.0% (6/10) — cost $1.12
- **RLM (gpt-5 + gpt-5-mini):** **70.0%** (7/10) — cost $0.28

### Per-item

| Item | Tokens | Diff | Base | RLM | RLM steps/bash/sub |
|------|--------|------|------|-----|--------------------|
| 66ebd3ba | 113K | hard | ❌ | ❌ | 9/8/0 |
| 66ec3644 | 99K  | hard | ❌ | ✅ | 19/18/0 |
| 66ecf139 | 49K  | hard | ✅ | ❌ | 13/12/0 |
| 66ed3e90 | 70K  | easy | ✅ | ✅ | 8/7/0 |
| 66f39ac5 | 86K  | easy | ❌ | ✅ | 13/12/0 |
| 66f3ad93 | 25K  | hard | ✅ | ✅ | 15/14/0 |
| 66f3c219 | 121K | hard | ✅ | ✅ | 18/17/0 |
| 66f3cb88 | 92K  | hard | ✅ | ✅ | 21/20/0 |
| 66f530ce | 48K  | easy | ✅ | ✅ | 10/9/0 |
| 66fa50ac | 82K  | easy | ❌ | ❌ | 14/13/0 |

Flips: RLM recovered 2 items the baseline missed (99K/hard, 86K/easy); baseline recovered 1 RLM missed (49K/hard). Net: +10 percentage points to RLM on this subset.

### vs. paper

| | Paper (full CodeQA, 23K–4.2M tokens) | Ours (≤128K subset) |
|---|---|---|
| GPT-5 baseline | 24% | 60% |
| GPT-5 RLM | 62% | 70% |
| Delta | **+38 pp** | **+10 pp** |

**Why our delta is smaller than the paper's:** we capped context at 128K so every baseline call *fits*. The paper's full suite includes items up to 4.2M tokens where baseline hits context overflow and scores 0% — which drags the baseline number down and inflates the gap. On items that fit the window, gpt-5 does well; RLM still wins modestly on accuracy and dramatically on cost.

Directionally consistent with the paper's claim: **RLM ≥ baseline on accuracy, and gets cheaper as context grows**.

## 3. Notes on sub-call usage

Across 22 RLM runs we observed **0 `llm` sub-calls** — gpt-5 solved everything by driving bash alone, no recursive sub-LM calls needed. The persistent REPL (shell vars + functions + files persisting across bash calls) was sufficient. Sub-calls would likely start firing on tasks that require true summarisation/aggregation (OOLONG-Pairs, BrowseComp-Plus deep-research), where a single grep hit doesn't answer the question.

## 4. What we did *not* run (and why)

| Benchmark | Paper result (GPT-5) | Why skipped |
|---|---|---|
| BrowseComp-Plus | 0% → **91.3%** | Full eval ~$1000; contexts 6M–11M tokens don't fit any single-model call anyway |
| OOLONG-Pairs | 0.04% → **58%** | Public dataset exists ([HF](https://huggingface.co/oolongbench)); would be the most dramatic demonstration but needs a careful replication pass — good next step |
| OOLONG (main) | 44% → **56.5%** | Feasible; similar cost to our CodeQA run — good next step |

## 5. Reproducing

```bash
pnpm tsx bench/run-niah.ts     # ~$1.50, ~5 min
pnpm tsx bench/run-codeqa.ts   # ~$1.50, ~15 min
pnpm tsx bench/summarize.ts    # aggregates bench/results/*.jsonl
```

Raw results in `bench/results/{niah,codeqa}.jsonl` (one record per run). Generators deterministic (seeded); CodeQA pulls items in `_id` sort order for reproducibility.

## 6. Caveats

- **N=3 / N=10 is small.** Confidence intervals are wide. Larger N would firm the CodeQA delta.
- **Cost accounting** charges RLM's sub-call tokens at the root-model rate (gpt-5) as a conservative upper bound. Real RLM cost is lower because sub-calls hit gpt-5-mini.
- **We do not reproduce the paper's judge-model setup** (OOLONG uses grading rubrics). Our scoring is exact-match for NIAH and letter extraction for CodeQA (matching LongBench's official eval).
- **Single run per condition.** No variance estimation across repeated runs on the same item.
