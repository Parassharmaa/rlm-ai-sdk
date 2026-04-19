/**
 * Run pairs V3 (NLP-embedded attributes) across 3 conditions.
 *
 *   pnpm tsx bench/run-pairs-v3.ts
 */
import { mkdirSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { generatePairsV3Item, scorePairsV3 } from "./pairs-v3.js";
import {
  logProgress,
  runBaseline,
  runRLMCondition,
  writeRecord,
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const N_ITEMS = 6;
const SEED_BASE = 2_000_000;

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
  outFile: "bench/results/pairs-v3.jsonl",
};
const cfgWithSub: RunConfig = {
  ...baseCfg,
  maxSteps: 40,
  maxSubCalls: 20,
  maxDepth: 0,
  outFile: "bench/results/pairs-v3.jsonl",
};

mkdirSync("bench/results", { recursive: true });
await writeFile(cfgNoSub.outFile, "");

console.log(`Pairs V3 (NLP-embedded): N=${N_ITEMS}, 3 conditions each.\n`);

let totalCost = 0;
for (let i = 0; i < N_ITEMS; i++) {
  const item = generatePairsV3Item(SEED_BASE + i);
  const tokens = Math.ceil(item.context.length / 4);

  // 1. Baseline
  {
    const r = await runBaseline(
      baseCfg.rootModelId,
      item.question,
      item.context,
      tokens,
      baseCfg.baselineContextLimitTokens,
    );
    const scored = r.error === "context_overflow" ? null : scorePairsV3(item, r.answer);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: item.id,
      condition: "baseline",
      score: scored === null ? null : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    console.log(`      gold=${item.answer} predicted=${scored?.predicted ?? "?"}`);
    totalCost += rec.costUSD;
  }
  // 2. RLM no-sub
  {
    const r = await runRLMCondition(cfgNoSub, item.question, item.context, tokens);
    const scored = r.error ? null : scorePairsV3(item, r.answer);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__no_sub`,
      condition: "rlm",
      score: scored === null ? 0 : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    console.log(`      gold=${item.answer} predicted=${scored?.predicted ?? "?"}`);
    totalCost += rec.costUSD;
  }
  // 3. RLM with-sub
  {
    const r = await runRLMCondition(cfgWithSub, item.question, item.context, tokens);
    const scored = r.error ? null : scorePairsV3(item, r.answer);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__with_sub`,
      condition: "rlm",
      score: scored === null ? 0 : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgWithSub, rec);
    logProgress(rec);
    console.log(`      gold=${item.answer} predicted=${scored?.predicted ?? "?"}`);
    totalCost += rec.costUSD;
  }
  console.log(`   running total: $${totalCost.toFixed(2)}\n`);
}

console.log(`\nTotal cost: $${totalCost.toFixed(2)}`);
console.log(`Results → ${cfgNoSub.outFile}`);
