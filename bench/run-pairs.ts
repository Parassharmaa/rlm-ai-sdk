/**
 * Run the synthetic pairs benchmark across 3 conditions.
 *
 *   pnpm tsx bench/run-pairs.ts
 */
import { mkdirSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { buildPrompt, generatePairsItem, scorePairs } from "./pairs.js";
import {
  logProgress,
  runBaseline,
  runRLMCondition,
  writeRecord,
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const N_ITEMS = 10;
const SEED_BASE = 1_000_000;

const baseCfg: Omit<RunConfig, "maxSteps" | "maxSubCalls" | "outFile"> = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  baselineContextLimitTokens: 400_000,
};

const cfgNoSub: RunConfig = {
  ...baseCfg,
  maxSteps: 30,
  maxSubCalls: 0,
  maxDepth: 0,
  outFile: "bench/results/pairs.jsonl",
};

const cfgWithSub: RunConfig = {
  ...baseCfg,
  maxSteps: 40,
  maxSubCalls: 20,
  maxDepth: 0,
  outFile: "bench/results/pairs.jsonl",
};

mkdirSync("bench/results", { recursive: true });
await writeFile(cfgNoSub.outFile, "");

console.log(`Pairs benchmark: N=${N_ITEMS}, 3 conditions each.\n`);

let totalCost = 0;
for (let i = 0; i < N_ITEMS; i++) {
  const item = generatePairsItem(SEED_BASE + i);
  const { query, context } = buildPrompt(item);
  const tokens = Math.ceil(context.length / 4);

  // 1. Baseline
  {
    const r = await runBaseline(
      baseCfg.rootModelId,
      query,
      context,
      tokens,
      baseCfg.baselineContextLimitTokens,
    );
    const scored = r.error === "context_overflow" ? null : scorePairs(item, r.answer);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: item.id,
      condition: "baseline",
      score: scored === null ? null : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    console.log(
      `      gold=${item.answer} predicted=${scored?.predicted ?? "?"}`,
    );
    totalCost += rec.costUSD;
  }
  // 2. RLM no-sub
  {
    const r = await runRLMCondition(cfgNoSub, query, context, tokens);
    const scored = r.error ? null : scorePairs(item, r.answer);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__no_sub`,
      condition: "rlm",
      score: scored === null ? 0 : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    console.log(
      `      gold=${item.answer} predicted=${scored?.predicted ?? "?"}`,
    );
    totalCost += rec.costUSD;
  }
  // 3. RLM with-sub
  {
    const r = await runRLMCondition(cfgWithSub, query, context, tokens);
    const scored = r.error ? null : scorePairs(item, r.answer);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__with_sub`,
      condition: "rlm",
      score: scored === null ? 0 : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgWithSub, rec);
    logProgress(rec);
    console.log(
      `      gold=${item.answer} predicted=${scored?.predicted ?? "?"}`,
    );
    totalCost += rec.costUSD;
  }
  console.log(`   running total: $${totalCost.toFixed(2)}\n`);
}

console.log(`\nTotal cost: $${totalCost.toFixed(2)}`);
console.log(`Results → ${cfgNoSub.outFile}`);
