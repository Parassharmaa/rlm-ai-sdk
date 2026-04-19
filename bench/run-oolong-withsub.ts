/**
 * Focused re-run: OOLONG with-sub only, with the tuned root prompt.
 * Produces a separate `oolong-v2.jsonl` so we can compare before/after.
 *
 *   pnpm tsx bench/run-oolong-withsub.ts
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
  runRLMCondition,
  writeRecord,
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const N_ITEMS = 10;

const cfg: RunConfig = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  maxSteps: 30,
  maxSubCalls: 10,
  maxDepth: 0,
  outFile: "bench/results/oolong-withsub-v2.jsonl",
  baselineContextLimitTokens: 400_000,
};

mkdirSync("bench/results", { recursive: true });
await writeFile(cfg.outFile, "");

const items = await fetchOolongCount32K({ n: N_ITEMS });
console.log(`OOLONG with-sub re-run on ${items.length} items (tuned prompt)\n`);

let totalCost = 0;
for (const item of items) {
  const { query, context } = buildPrompt(item);
  const labels = labelsFromQuestion(item.question);
  const tokens = Math.ceil(context.length / 4);

  const r = await runRLMCondition(cfg, query, context, tokens);
  const rec: RunRecord = {
    suite: "oolong",
    itemId: String(item.id) + "__with_sub_v2",
    condition: "rlm",
    score: r.error ? 0 : scoreOolong(item, labels, r.answer).correct ? 1 : 0,
    ...r,
  };
  writeRecord(cfg, rec);
  logProgress(rec);
  totalCost += rec.costUSD;
  console.log(`   running total: $${totalCost.toFixed(2)}`);
}

console.log(`\nTotal cost: $${totalCost.toFixed(2)}`);
console.log(`Results → ${cfg.outFile}`);
