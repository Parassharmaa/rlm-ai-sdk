/**
 * OOLONG counting @ 32K with gpt-5-mini as the ROOT (not gpt-5).
 *
 * Hypothesis: the paper's sub-call advantage should show up when the root
 * LM is too weak to solve the task bash-only. With gpt-5 root, our bench
 * suite shows sub-calls never fire because the root is strong enough.
 * Swapping to gpt-5-mini root tests the core RLM thesis more faithfully.
 *
 *   pnpm tsx bench/run-oolong-mini-root.ts
 */
import { mkdirSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import {
  buildPrompt,
  fetchOolongCounting,
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

const N_ITEMS = 6;

const baseCfg: Omit<RunConfig, "maxSteps" | "maxSubCalls" | "outFile"> = {
  rootModelId: "gpt-5-mini", // ← weaker root
  subModelId: "gpt-5-mini",  // same model for sub (cheapest test)
  baselineContextLimitTokens: 200_000,
};

const cfgNoSub: RunConfig = {
  ...baseCfg,
  maxSteps: 30,
  maxSubCalls: 0,
  maxDepth: 0,
  outFile: "bench/results/oolong-mini-root.jsonl",
};

const cfgWithSub: RunConfig = {
  ...baseCfg,
  maxSteps: 30,
  maxSubCalls: 10,
  maxDepth: 0,
  outFile: "bench/results/oolong-mini-root.jsonl",
};

mkdirSync("bench/results", { recursive: true });
await writeFile(cfgNoSub.outFile, "");

const items = await fetchOolongCounting({ n: N_ITEMS, contextLen: 32768 });
console.log(
  `OOLONG 32K with gpt-5-mini as ROOT, N=${items.length}, 3 conditions\n`,
);

let totalCost = 0;
for (const item of items) {
  const { query, context } = buildPrompt(item);
  const labels = labelsFromQuestion(item.question);
  const tokens = Math.ceil(context.length / 4);

  // 1. Baseline (gpt-5-mini direct)
  {
    const r = await runBaseline(
      baseCfg.rootModelId,
      query,
      context,
      tokens,
      baseCfg.baselineContextLimitTokens,
    );
    const scored = r.error === "context_overflow" ? null : scoreOolong(item, labels, r.answer);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__mini_base`,
      condition: "baseline",
      score: scored === null ? null : scored.correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  // 2. RLM no-sub (gpt-5-mini root, bash only)
  {
    const r = await runRLMCondition(cfgNoSub, query, context, tokens);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__mini_no_sub`,
      condition: "rlm",
      score: r.error ? 0 : scoreOolong(item, labels, r.answer).correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  // 3. RLM with-sub (gpt-5-mini root + sub)
  {
    const r = await runRLMCondition(cfgWithSub, query, context, tokens);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__mini_with_sub`,
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
