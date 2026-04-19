/**
 * OOLONG counting at 131K tokens — the paper's actual scale for the
 * OOLONG (main) benchmark. Baseline vs RLM no-sub only (sub-calls
 * already validated as not helpful on simple counting).
 *
 * Paper's GPT-5 result at 131K: baseline 44% → RLM 56.5% (+12.5 pp).
 *
 *   pnpm tsx bench/run-oolong-131k.ts
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
const CTX_LEN = 131072;

const baseCfg: Omit<RunConfig, "maxSteps" | "maxSubCalls" | "outFile"> = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  baselineContextLimitTokens: 400_000,
};

const cfgNoSub: RunConfig = {
  ...baseCfg,
  maxSteps: 40,
  maxSubCalls: 0,
  maxDepth: 0,
  outFile: "bench/results/oolong-131k.jsonl",
};

mkdirSync("bench/results", { recursive: true });
await writeFile(cfgNoSub.outFile, "");

console.log(`Fetching ${N_ITEMS} OOLONG counting@${CTX_LEN} items...`);
const items = await fetchOolongCounting({ n: N_ITEMS, contextLen: CTX_LEN });
console.log(`Got ${items.length} items. Running 2 conditions each.\n`);

let totalCost = 0;
for (const item of items) {
  const { query, context } = buildPrompt(item);
  const labels = labelsFromQuestion(item.question);
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
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__131k`,
      condition: "baseline",
      score:
        r.error === "context_overflow"
          ? null
          : scoreOolong(item, labels, r.answer).correct
            ? 1
            : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  // 2. RLM no-sub
  {
    const r = await runRLMCondition(cfgNoSub, query, context, tokens);
    const rec: RunRecord = {
      suite: "oolong",
      itemId: `${item.id}__131k_no_sub`,
      condition: "rlm",
      score: r.error ? 0 : scoreOolong(item, labels, r.answer).correct ? 1 : 0,
      ...r,
    };
    writeRecord(cfgNoSub, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  console.log(`   running total: $${totalCost.toFixed(2)}`);
}

console.log(`\nTotal cost: $${totalCost.toFixed(2)}`);
console.log(`Results → ${cfgNoSub.outFile}`);
