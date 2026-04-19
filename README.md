# rlm-ai-sdk

[![CI](https://github.com/Parassharmaa/rlm-ai-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Parassharmaa/rlm-ai-sdk/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Recursive Language Models (RLM) for the Vercel AI SDK.** Based on [Zhang, Kraska & Khattab (2025)](https://arxiv.org/abs/2512.24601), adapted with a **persistent bash REPL** instead of the paper's Python/Jupyter.

The context never enters the LM's prompt — it lives as files in a sandbox. The root LM drives `bash` (to grep/slice) and optionally `llm` (to recurse on focused snippets), then calls `final(answer)`.

## Install (from GitHub)

Not on npm yet. Install directly from this repo:

```bash
pnpm add github:Parassharmaa/rlm-ai-sdk ai zod @ai-sdk/openai
```

Requires Node ≥ 24 and `bash` on your PATH.

## Usage

### Direct

```ts
import { openai } from "@ai-sdk/openai";
import { runRLM } from "rlm-ai-sdk";

const result = await runRLM(
  { model: openai("gpt-5-mini") },
  { query: "Who is the CEO?", context: hugeChunksArray },
);
console.log(result.answer, result.usage, result.trace);
```

### Middleware (auto-route oversized prompts)

```ts
import { wrapLanguageModel, generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { createRLMMiddleware } from "rlm-ai-sdk";

const model = wrapLanguageModel({
  model: openai("gpt-5-mini"),
  middleware: createRLMMiddleware({ thresholdTokens: 20_000 }),
});
const { text } = await generateText({ model, prompt: "..." });
```

Small prompts pass through; large prompts divert to RLM. Both `generateText` and `streamText` are supported.

### Agent tool

```ts
import { generateText, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";
import { rlmTool } from "rlm-ai-sdk";

const model = openai("gpt-5-mini");
await generateText({
  model, tools: { rlm: rlmTool({ model }) },
  prompt: "Find the launch code in: …", stopWhen: stepCountIs(6),
});
```

## Configuration

```ts
interface RLMEngineConfig {
  model: LanguageModel;          // root LM (required)
  subModel?: LanguageModel;      // cheaper model for sub-calls (recommended)
  maxSteps?: number;             // default 40
  subMaxSteps?: number;          // default 20 (when maxDepth ≥ 1)
  maxSubCalls?: number;          // default 20, across the whole recursion tree
  maxDepth?: number;             // 0 (default) = leaf sub-calls; 1+ = nested RLMs
  bashTimeoutMs?: number;        // default 20_000
  bashOutputByteCap?: number;    // default 8_192 — combined stdout+stderr
  subPromptCharCap?: number;     // default 400_000
  sandboxRoot?: string;          // defaults to os.tmpdir()
  onEvent?: (e: RLMEvent) => void;
  signal?: AbortSignal;
}
```

## Benchmarks (GPT-5 root + GPT-5-mini sub)

| Benchmark | Baseline | RLM | Cost |
|---|---|---|---|
| **CodeQA 136K–483K** (N=10) | **30%** | **90%** | 5× cheaper &nbsp;·&nbsp; **+60 pp** |
| OOLONG counting 32K (N=10) | 60% | **90%** | 3× cheaper &nbsp;·&nbsp; +30 pp |
| S-NIAH 8K–256K | 100% | 100% | **93× cheaper** @256K |
| CodeQA ≤128K pooled (N=35) | 62.9% | 62.9% | 4× cheaper, accuracy tie |
| OOLONG counting 131K (N=6) | 67% | 67% | 4.6× cheaper |
| Synthetic pairs 45K (N=10) | 100% | 90–100% | 14× cheaper |
| Pairs V3 NLP 8K (N=6) | 100% | 67% | 2× cheaper — baseline wins |

**Headline:** RLM is 2–93× cheaper everywhere, and wins accuracy decisively (+60 pp) when context overflows the model's window — reproducing the paper's central CodeQA claim. On tasks that fit the window, accuracy is roughly equivalent.

Full methodology, per-item results, caveats: [`bench/results.md`](./bench/results.md).

## When to use RLM

**Use it when:** context routinely exceeds the model's window; you're aggregating over many items; you want context-size-independent cost.

**Skip it when:** the whole context fits the window and GPT-5 can answer in one shot; the task is shallow fact-extraction; you need real token-by-token streaming.

Defaults are tuned from measurements: leave `maxDepth: 0` unless you *know* the task needs per-item NLP extraction. Sub-calls fired once in ~100 runs across our suite — capable models prefer bash.

## Sandbox

One persistent `bash` process per invocation; shell state persists across `bash` calls. Stderr merged into stdout. Per-call timeouts (SIGKILL). Credential-like env vars (`*_API_KEY`, `*_SECRET`, `*_TOKEN`, `AWS_*`, etc.) blanked before the child sees them.

> **Not a security boundary.** bash runs at host UID. For adversarial contexts, wrap in a container or `bwrap`/`firejail`.

## Development

```bash
git clone https://github.com/Parassharmaa/rlm-ai-sdk.git
cd rlm-ai-sdk && pnpm install && pnpm check   # typecheck + 37 tests + build

pnpm example:basic        # smoke against real OpenAI (reads .env)
pnpm example:middleware
pnpm example:tool
```

Benchmarks under `bench/` — see [`bench/results.md`](./bench/results.md) for the reproduction script block (~$20, ~2 h for the full suite).

## License

MIT
