/**
 * CodeQA at > 128K tokens — the paper's "baseline overflows, RLM wins" scenario.
 *
 * We pick 10 items in the 136K–500K token range. GPT-5's input window is
 * ~400K tokens, so:
 *   - items ≤400K: baseline should fit (may still struggle at high sizes);
 *   - items >400K: baseline skips as `context_overflow`, RLM handles them.
 *
 * This is the regime the paper's +38 pp CodeQA gap comes from. Our earlier
 * runs capped at ≤128K and found a tie; this run tests the other side.
 *
 *   pnpm tsx bench/run-codeqa-large.ts
 */
import { mkdirSync, readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { buildPrompt, scoreCodeQA } from "./longbench.js";
import type { LongBenchItem } from "./longbench.js";
import {
  logProgress,
  runBaseline,
  runRLMCondition,
  writeRecord,
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const N_ITEMS = 10;
const MIN_TOKENS = 128_000;
const MAX_TOKENS = 500_000;

const baseCfg: Omit<RunConfig, "maxSteps" | "maxSubCalls" | "outFile"> = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  // GPT-5 docs state 400K tokens input. Leave a small margin for system
  // prompt + user instructions.
  baselineContextLimitTokens: 395_000,
};

const cfg: RunConfig = {
  ...baseCfg,
  maxSteps: 40,
  maxSubCalls: 0,
  maxDepth: 0,
  outFile: "bench/results/codeqa-large.jsonl",
};

const all = JSON.parse(
  readFileSync("bench/data/longbench-v2.json", "utf8"),
) as LongBenchItem[];
const items = all
  .filter((r) => r.domain === "Code Repository Understanding")
  .filter(
    (r) =>
      r.context.length >= MIN_TOKENS * 4 &&
      r.context.length <= MAX_TOKENS * 4,
  )
  .sort((a, b) => a.context.length - b.context.length)
  .slice(0, N_ITEMS);

mkdirSync("bench/results", { recursive: true });
await writeFile(cfg.outFile, "");

console.log(
  `CodeQA large subset (${MIN_TOKENS / 1000}K–${MAX_TOKENS / 1000}K tokens, N=${items.length})\n`,
);

let totalCost = 0;
for (const item of items) {
  const { query, context } = buildPrompt(item);
  const tokens = Math.ceil(context.length / 4);

  // Baseline
  {
    const r = await runBaseline(
      cfg.rootModelId,
      query,
      context,
      tokens,
      cfg.baselineContextLimitTokens,
    );
    const scored = r.error === "context_overflow" ? null : scoreCodeQA(item, r.answer);
    const rec: RunRecord = {
      suite: "codeqa",
      itemId: item._id,
      condition: "baseline",
      score: scored === null ? null : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfg, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  // RLM
  {
    const r = await runRLMCondition(cfg, query, context, tokens);
    const scored = r.error ? null : scoreCodeQA(item, r.answer);
    const rec: RunRecord = {
      suite: "codeqa",
      itemId: item._id,
      condition: "rlm",
      score: scored === null ? 0 : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfg, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  console.log(`   running total: $${totalCost.toFixed(2)}`);
}

console.log(`\nTotal cost: $${totalCost.toFixed(2)}`);
console.log(`Results → ${cfg.outFile}`);
