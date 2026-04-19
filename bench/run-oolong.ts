/**
 * OOLONG counting @ 32K — A/B/C across 3 conditions.
 *
 * Tests whether the recursion path (maxDepth=1, sub-RLMs) actually helps
 * vs. the "no sub-calls" ablation and vs. baseline direct generateText.
 *
 *   pnpm tsx bench/run-oolong.ts
 */
import { mkdirSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import {
  buildPrompt,
  fetchOolongCount32K,
  labelsFromQuestion,
  scoreOolong,
} from "./oolong.js";
import {
  logProgress,
  runBaseline,
  runRLMCondition,
  writeRecord,
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const N_ITEMS = 10;

const baseCfg: Omit<RunConfig, "maxSteps" | "maxSubCalls" | "outFile"> = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  baselineContextLimitTokens: 400_000,
};

const cfgNoSub: RunConfig = {
  ...baseCfg,
  maxSteps: 30,
  maxSubCalls: 0, // ablation: budget 0 ⇒ every llm() returns an error (bash only)
  maxDepth: 0,
  outFile: "bench/results/oolong.jsonl",
};

const cfgWithSub: RunConfig = {
  ...baseCfg,
  maxSteps: 30,
  maxSubCalls: 10, // sub-calls are LEAVES (maxDepth=0) — matches paper's llm_query
  maxDepth: 0,
  outFile: "bench/results/oolong.jsonl",
};

mkdirSync("bench/results", { recursive: true });
await writeFile(cfgNoSub.outFile, "");

console.log(`Fetching ${N_ITEMS} OOLONG counting@32K items...`);
const items = await fetchOolongCount32K({ n: N_ITEMS });
console.log(`Got ${items.length} items. Running 3 conditions each.\n`);

let totalCost = 0;
for (const item of items) {
  const { query, context } = buildPrompt(item);
  const labels = labelsFromQuestion(item.question);
  const tokens = Math.ceil(context.length / 4);

  const scoreAnswer = (r: { error: string | null; answer: string }) =>
    r.error === "context_overflow"
      ? null
      : scoreOolong(item, labels, r.answer).correct
        ? 1
        : 0;

  // 1. Baseline
  {
    const r = await runBaseline(
      baseCfg.rootModelId,
      query,
      context,
      tokens,
      baseCfg.baselineContextLimitTokens,
    );
    const rec: RunRecord = {
      suite: "oolong",
      itemId: String(item.id),
      condition: "baseline",
      score: scoreAnswer(r),
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  // 2. RLM w/o sub-calls (maxSubCalls=0, maxDepth=0) — the paper's ablation
  {
    const r = await runRLMCondition(cfgNoSub, query, context, tokens);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: String(item.id) + "__no_sub",
      condition: "rlm",
      score: r.error ? 0 : scoreOolong(item, labels, r.answer).correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  // 3. RLM w/ sub-calls (maxSubCalls=20, maxDepth=1)
  {
    const r = await runRLMCondition(cfgWithSub, query, context, tokens);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: String(item.id) + "__with_sub",
      condition: "rlm",
      score: r.error ? 0 : scoreOolong(item, labels, r.answer).correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgWithSub, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  console.log(`   running total: $${totalCost.toFixed(2)}`);
}

console.log(`\nTotal cost: $${totalCost.toFixed(2)}`);
console.log(`Results → ${cfgNoSub.outFile}`);
