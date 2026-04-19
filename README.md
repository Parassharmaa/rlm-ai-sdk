# rlm-ai-sdk

**Recursive Language Models (RLM) for the Vercel AI SDK.**

Give any AI SDK language model a bash-sandboxed REPL and a recursive `llm` call so it can programmatically explore contexts that are orders of magnitude larger than its attention window.

Based on Zhang, Kraska & Khattab, [*Recursive Language Models*](https://arxiv.org/abs/2512.24601) (2025), adapted to TypeScript and the AI SDK — with a **persistent bash REPL** in place of the paper's Python/Jupyter REPL. Shell variables, functions, and cwd persist across bash calls so the root LM can build up state (found regions, chunk lists, intermediate buffers) instead of re-deriving it every step.

## Why

Long-context LLMs suffer from *context rot* — the model's answer quality degrades as the prompt grows, even when the relevant information is technically in-window. RLM sidesteps this by **not putting the large context in the prompt at all**. Instead:

1. The "root" LM sees only a short instruction plus a catalog of what's available.
2. The large context lives as files in a sandboxed workdir.
3. The root LM drives a loop of `bash` (to inspect/slice) and `llm` (to recurse on focused snippets).
4. When done, it emits a `final(answer)` tool call.

## Install

```bash
pnpm add rlm-ai-sdk ai zod
# and at least one provider
pnpm add @ai-sdk/openai
```

Requirements: Node ≥ 24, and `bash` on your PATH.

## Three ways to use it

### 1. Direct — `runRLM`

```ts
import { openai } from "@ai-sdk/openai";
import { runRLM } from "rlm-ai-sdk";

const result = await runRLM(
  { model: openai("gpt-5-mini") },
  {
    query: "Who is the CEO of Acme Corp?",
    context: hugeChunksArray,   // string | string[] | ContextItem[]
  },
);

console.log(result.answer);
console.log(result.trace);       // every bash/sub-llm/final event
```

### 2. Middleware — transparently route oversized prompts

```ts
import { wrapLanguageModel } from "ai";
import { openai } from "@ai-sdk/openai";
import { createRLMMiddleware } from "rlm-ai-sdk";

const model = wrapLanguageModel({
  model: openai("gpt-5-mini"),
  middleware: createRLMMiddleware({ thresholdTokens: 20_000 }),
});

// Small calls → pass through. Large calls → RLM. Same surface to callers.
const { text } = await generateText({ model, prompt: "..." });
```

### 3. Tool — give an agent an `rlm` tool to recurse on demand

```ts
import { generateText, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";
import { rlmTool } from "rlm-ai-sdk";

const model = openai("gpt-5-mini");
const result = await generateText({
  model,
  prompt: "Find the launch code in this log: ...",
  tools: { rlm: rlmTool({ model }) },
  stopWhen: stepCountIs(6),
});
```

## How the loop works

```
   ┌───────────────────────────────┐
   │           Root LM             │
   │  (system: ROOT_SYSTEM_PROMPT) │
   └───────────┬───────────────────┘
               │ tool calls
   ┌───────────┴───────────┬──────────────────┬──────────────┐
   │                       │                  │              │
   ▼                       ▼                  ▼              ▼
 bash(cmd)             llm(prompt)        final(answer)    (repeat)
   │                       │                  │
   ▼                       ▼                  ▼
 bash -c  in              sub-LM           stop the
 sandbox workdir       generateText        loop, return
 (ctx files expos-     with SUB_SYSTEM_
  ed as $RLM_CTX_*)    PROMPT
```

Strategies the root LM picks up naturally: **peek** (`wc -c`, `head`), **grep** (`grep -n`), **partition + map** (`split -l`, per-chunk `llm` calls), **summarize then dive**.

## Configuration

```ts
interface RLMEngineConfig {
  model: LanguageModel;          // required — root LM
  subModel?: LanguageModel;      // optional — used for sub-calls, defaults to model
  maxSteps?: number;             // default 40
  maxSubCalls?: number;          // default 20
  bashTimeoutMs?: number;        // default 20_000
  bashOutputByteCap?: number;    // default 8_192
  subPromptCharCap?: number;     // default 400_000
  sandboxRoot?: string;          // defaults to os.tmpdir()
  onEvent?: (e: RLMEvent) => void;
  signal?: AbortSignal;
}
```

## Sandbox contract

- One **persistent bash process** per invocation — shell variables, functions, aliases, and cwd persist across `execute` calls. Disposed on completion or error.
- Each invocation gets a fresh temp workdir; context items written as `<id>.txt` into `$RLM_CTX_DIR`, each also exposed via `$RLM_CTX_<ID>`.
- stderr is merged into stdout at startup via `exec 2>&1` — you get one combined byte stream per call, capped at `bashOutputByteCap`.
- Calls are **serialized**; the REPL is single-threaded.
- On timeout: the bash is SIGKILLed and the sandbox is marked dead. Subsequent calls return a dead-result (non-zero exit, `stderr` explaining the cause). The RLM loop treats this as recoverable — the model can still call `final`.
- The user issuing `exit` inside the REPL will likewise kill the shell and is treated the same way. Shell protocol identifiers (`__rlm_*`) are reserved; the root prompt tells the model not to touch them.
- Env vars whose names match common credential patterns (`*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`, `*_CREDENTIALS`, and `AWS_*` / `AZURE_*` / `GCP_*` prefixes) are blanked in the child's environment. This is a best-effort filter — if you set non-standard secret names, blank them yourself before invoking the sandbox.

> **Security note:** this is a convenience sandbox, not a security boundary. `bash -c` runs with the same UID as the host process. For untrusted or adversarial contexts, wrap `BashSandbox` in a container, VM, or `bwrap`/`firejail` layer.

## Development

```bash
pnpm install          # install
pnpm build            # tsup → dist (esm + cjs + dts)
pnpm test             # vitest (unit)
pnpm typecheck        # tsc --noEmit
pnpm example:basic    # e2e smoke with real OpenAI
pnpm example:middleware
pnpm example:tool
```

The examples read `.env` at the repo root for `OPENAI_API_KEY`.

## Streaming

`streamText({ model: wrapped })` works: when a call is diverted to RLM, the middleware runs the full RLM loop synchronously, then emits a synthetic `text-start` → `text-delta` (the full answer) → `text-end` → `finish` stream. You get the same UI integration points as a real stream, but the answer arrives in one chunk — streaming token-by-token during the RLM loop itself isn't meaningful (the loop runs many sub-calls and only the final answer is visible).

## Token usage

`RLMResult.usage` aggregates `inputTokens`, `outputTokens`, and `totalTokens` across the root LM and every sub-call, where the underlying provider reports them (all major providers do). The middleware also surfaces totals via `providerMetadata["rlm-ai-sdk"]` on the `generateText`/`streamText` result.

## Benchmarks

Benchmarked with the same models as the [RLM paper](https://arxiv.org/abs/2512.24601): **GPT-5** (root) + **GPT-5-mini** (sub-calls).

| Benchmark | Context | Baseline | RLM (bash only) | RLM + sub-calls | Cost |
|---|---|---|---|---|---|
| S-NIAH | 8K–256K | 100% | 100% | — | RLM **93× cheaper** @256K |
| LongBench-v2 CodeQA (pooled N=35) | 25K–121K | 62.9% | 62.9% | — | RLM **4× cheaper** (stable) |
| OOLONG counting @ 32K (N=10) | 24K | 60% | **90%** | 80%† | RLM no-sub **3× cheaper** |
| OOLONG counting @ 131K (N=6) | 96K | 67% | 67% | — | RLM **4.6× cheaper** |
| Synthetic pairs @ 45K (N=10) | 45K | 100% | 90–100% | 100% | RLM **14× cheaper** |
| Synthetic pairs V3 NLP @ 8K (N=6) | 8.5K | **100%** | 67% | 67% | RLM ~2× cheaper |
| OOLONG 32K w/ **gpt-5-mini** root (N=6) | 24K | 67% | 67% | **83%**‡ | RLM 2–3× cheaper |
| **CodeQA LARGE** (136K–483K, N=10) | up to 483K | **30%** | **90%** | — | **+60 pp RLM**, 5× cheaper |

<sup>‡ First observed sub-call use: 1 call across 6 items. Still N=6 small-sample — the +1 item accuracy came from `sub=0` sampling variance on a different item, not from the sub-call. Noisy but directionally consistent.</sup>

<sup>† With tuned root prompt. Untuned: 70%. RLM's 30 pp advantage at 32K collapses to a tie at 131K — gpt-5 handles long contexts better than the paper assumed.</sup>

**Honest findings across eight benchmarks:**
- **Cost is the stable RLM win.** 2×–93× cheaper depending on context size. Never underwater.
- **RLM wins accuracy decisively when context overflows the model's window** (CodeQA LARGE, N=10: 30% → 90%, +60 pp). This reproduces the paper's central claim for LongBench-v2 CodeQA.
- **Accuracy ties or slightly loses on tasks that fit the window** — once baseline can read everything, RLM's grep-based exploration is roughly equivalent.
- **Sub-calls fire very rarely** — 1 observed call in ~100+ RLM runs across our suite. GPT-5 prefers bash pattern-matching over delegation. The paper's sub-call win requires a task where per-item extraction can't reduce to text processing *and* context is too big for baseline — we haven't cleanly reproduced that combination.

Full per-item results, methodology, and caveats: [`bench/results.md`](./bench/results.md).

## When to use RLM (prescriptive, from our benchmarks)

**Reach for `runRLM` / the middleware when any of these hold:**

1. **Your context routinely exceeds the model's window, or is close to the cost limit.** This is the 93× cost win at 256K. If your prompts are ≤20K tokens, vanilla `generateText` is usually fine.
2. **You're doing aggregation over many items** (counting, grouping, statistics over a long list of small records). OOLONG 32K showed +30 pp accuracy for RLM over baseline GPT-5 in exactly this regime.
3. **You want predictable cost regardless of context size.** RLM's cost scales with how much the *root LM* reads, not with total context — so a 1 MB document costs roughly the same as a 32K one if both tasks involve grepping a section.

**Don't reach for RLM when:**

1. **The whole context fits comfortably in the window AND doesn't need aggregation.** GPT-5 at <100K tokens is accurate enough that RLM's grep-based exploration is pure overhead in latency (+5–10s per call).
2. **Per-item reasoning is shallow.** If your task is "extract the one fact from a long doc," baseline does it faster and as accurately.
3. **You need streamed, token-by-token output.** RLM's `streamText` path emits one synthetic chunk with the final answer — not real streaming.

**Config guidance (based on measured behaviour):**

- `maxDepth: 0` (default). Our benchmarks showed sub-calls fire rarely and add wall-time when they do. Leave this unless you're *certain* the task has per-item NLP extraction over hundreds of items.
- `maxSubCalls: 0–10`. Gives the model a ceiling without encouraging it; the root prompt already biases against over-delegation.
- `maxSteps: 30–40`. Our runs converge in 5–20 steps for successful cases; failures tend to be step-budget exhaustion, which happens around step 25+.
- `bashOutputByteCap`: the default 8 KB is enough for most `head`/`grep -n` output. Increase if you see `TRUNCATED` in traces.
- **`subModel`: set to a cheaper model than `model`.** Matches the paper's GPT-5 + GPT-5-mini pairing; keeps sub-call cost low on the rare occasions they fire.

## Reproducing the benchmarks

```bash
# Total ~$20 for the full suite with gpt-5 / gpt-5-mini (2+ hours)
bash bench/download-data.sh              # fetch LongBench-v2 from HF (~465 MB)
pnpm tsx bench/run-niah.ts               # S-NIAH sweep      (~$1.50, 5 min)
pnpm tsx bench/run-codeqa.ts             # CodeQA N=15       (~$2.00, 20 min)
pnpm tsx bench/run-oolong.ts             # OOLONG 32K 3-cond (~$4.00, 90 min)
pnpm tsx bench/run-oolong-131k.ts        # OOLONG 131K       (~$1.50, 15 min)
pnpm tsx bench/run-pairs.ts              # synthetic pairs   (~$1.60, 15 min)
pnpm tsx bench/run-pairs-v3.ts           # pairs NLP variant (~$0.85, 15 min)
pnpm tsx bench/run-oolong-mini-root.ts   # mini as root      (~$0.20, 15 min)
pnpm tsx bench/summarize.ts              # aggregate all JSONL → markdown
```

## Limitations

- The middleware's query/context extraction is heuristic (last user message = query, everything prior = context). If extraction fails (e.g. no user message at all), the call silently falls through to the underlying model. For complex message histories, prefer `runRLM` or `rlmTool` directly.
- The sandbox is a convenience boundary, not a security one: bash runs at host UID. For adversarial contexts, wrap in a container or `bwrap`/`firejail`.

## License

MIT
