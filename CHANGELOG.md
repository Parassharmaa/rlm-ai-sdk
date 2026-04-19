# Changelog

All notable changes to `rlm-ai-sdk`. This project started life at commit
`52dde15` on 2026-04-15 and has evolved through benchmark-driven iteration.

## [Unreleased] — 2026-04-19

### Benchmarks (honest summary after 7 runs, ~$20 spend)

RLM wins on **cost** (2×–93× cheaper across every benchmark). Accuracy is
task-dependent: wins on small-scale aggregation, ties on most tasks, loses on
tasks that fit comfortably in the window and don't need aggregation.

| Benchmark | Baseline | RLM bash-only | Notes |
|---|---|---|---|
| S-NIAH 8K–256K | 100% | 100% | 93× cheaper at 256K |
| LongBench-v2 CodeQA (pooled N=35) | 62.9% | 62.9% | Statistical tie, 4× cheaper |
| OOLONG counting @ 32K (N=10) | 60% | **90%** | RLM wins aggregation, 3× cheaper |
| OOLONG counting @ 131K (N=6) | 67% | 67% | RLM advantage collapses at paper's scale |
| Synthetic pairs @ 45K (N=10) | 100% | 90–100% | **14× cheaper** |
| Synthetic pairs V3 NLP @ 8.5K (N=6) | **100%** | 67% | Baseline wins on small NLP tasks |
| OOLONG 32K with **gpt-5-mini** root (N=6) | 67% | 67%–83% | First observed sub-call use (1) |

Sub-call path: fired **1 time in ~96 RLM runs**. With capable root models
(GPT-5, even GPT-5-mini) bash exploration is usually sufficient. This
informs the default `maxDepth: 0`.

## 0.2.0 — true RLM recursion (commits `2a4a6f3` → `1b5bba0`)

### Added

- **True RLM recursion** (`2a4a6f3`): the `llm` tool can now re-enter the
  engine at `depth+1` with its own bash sandbox. Below `maxDepth`, sub-calls
  are themselves RLMs; at depth, they fall back to plain `generateText`.
- `maxDepth` and `subMaxSteps` config options (default `maxDepth=0` —
  sub-calls are leaves, matching the paper's `llm_query` semantics).
- `RLMEvent` variants: `sub-start` / `sub-end` for nested invocations;
  `sub-llm` is now leaf-only.
- Shared state threads budgets (`maxSubCalls`, usage counters) across the
  entire recursion tree so depth-2 calls can't multiply resources.
- Benchmark harness under `bench/`:
  - `bench/niah.ts` — S-NIAH synthetic needle-in-haystack.
  - `bench/longbench.ts` — LongBench-v2 CodeQA loader (Hugging Face).
  - `bench/oolong.ts` — OOLONG counting loader, parameterised `contextLen`.
  - `bench/pairs.ts` + `bench/pairs-v3.ts` — synthetic quadratic tasks.
  - `bench/runner.ts` — A/B runner with cost guardrails.
  - `bench/summarize.ts` — aggregates JSONL → markdown table.
- **Tuned root prompt** (`e24a529`): explicit "when NOT to call `llm`" rules,
  reordered strategies (bash-first), "finish in first 2/3 of budget" hint.
  Lifted OOLONG with-sub from 70% → 80% and dropped wall time 43%.
- Dataset loaders with retry + rate-limit handling for HF datasets-server.
- `README.md`: "When to use RLM" prescriptive guide + reproduction scripts.

### Changed

- **Breaking: `llm` tool signature changed from `{ prompt }` to
  `{ query, context }`** (`2a4a6f3`). Explicit split so the nested RLM
  knows what's the sub-question vs the context slice. Consumers who were
  calling the tool from scripted models need to update.
- Nested sub-RLMs use `subModel` when configured (matches the paper's
  GPT-5 + GPT-5-mini pairing). Root still uses `model`.
- Default `maxDepth` from `1` → `0` (`0ffc455`). A 1-item smoke with nested
  recursion burned $0.28 and 19 min without finalising on a task that leaf
  sub-call solved in 2 min. Nesting is opt-in.
- `SUB_SYSTEM_PROMPT` renamed to `SUB_LEAF_SYSTEM_PROMPT` (old name kept as
  deprecated alias).
- Examples rewritten (`caab6bc`) to use real scenarios (the package's own
  source code) rather than synthetic filler, with cost display and trace.

### Fixed

- (`2a4a6f3`) `steps` counter now reflects root-step count only; sub-call
  steps are tracked separately via `subCalls`.

## 0.1.0 — initial release (`52dde15`)

- Persistent bash REPL sandbox (`src/sandbox.ts`). One `bash` process per
  invocation; shell state persists across tool calls; credential-like env
  vars blanked via regex.
- `runRLM` / `RLMEngine` core loop: root LM drives tools `bash`, `llm`,
  `final`. Stops on `final` or `maxSteps`.
- `createRLMMiddleware` — `wrapLanguageModel` middleware that auto-routes
  oversized prompts through RLM; both `wrapGenerate` and `wrapStream`
  paths.
- `rlmTool` — standard AI SDK tool an agent can call on demand.
- 33 unit tests, 1 opt-in e2e test (`tests/e2e/needle.test.ts`).
- Three runnable examples: `basic.ts`, `middleware.ts`, `tool.ts`.
- Node ≥24, TypeScript 6.0, Vitest 4, Zod 4, AI SDK 6.

## Benchmark methodology

All numbers in this file come from real runs against the OpenAI API using
`gpt-5` as root and `gpt-5-mini` as sub-model (the paper's primary setup).
Raw per-item records are saved to `bench/results/*.jsonl` and aggregated
into `bench/results.md` by `bench/summarize.ts`. Prices in `bench/runner.ts`
reflect OpenAI's published rates as of 2026-04.

Sample sizes are small (N=6–15 per benchmark) for budget reasons; any
single flipped item swings accuracy by ~10 pp. Treat every delta below
15 pp as "within noise" unless aggregated across multiple runs.
