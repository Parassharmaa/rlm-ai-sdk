/**
 * Run LongBench-v2 CodeQA subset (≤128K tokens) with GPT-5 / GPT-5-mini.
 *
 *   pnpm tsx bench/run-codeqa.ts
 */
import { mkdirSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { buildPrompt, loadCodeQA, scoreCodeQA } from "./longbench.js";
import {
  logProgress,
  runBaseline,
  runRLMCondition,
  writeRecord,
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const N_ITEMS = 10;
const MAX_TOKENS = 128_000;

const cfg: RunConfig = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  maxSteps: 40,
  maxSubCalls: 15,
  outFile: "bench/results/codeqa.jsonl",
  baselineContextLimitTokens: 400_000,
};

mkdirSync("bench/results", { recursive: true });
await writeFile(cfg.outFile, "");

const items = loadCodeQA({ maxContextTokens: MAX_TOKENS, limit: N_ITEMS });
console.log(`CodeQA subset: ${items.length} items up to ${MAX_TOKENS} tokens`);

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
