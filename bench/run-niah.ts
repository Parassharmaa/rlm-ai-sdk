/**
 * Run the S-NIAH sweep with GPT-5 root / GPT-5-mini sub (paper setup).
 *
 *   pnpm tsx bench/run-niah.ts
 */
import { mkdirSync, existsSync } from "node:fs";
import { buildNIAHSweep, scoreNIAH } from "./niah.js";
import {
  logProgress,
  runBaseline,
  runRLMCondition,
  writeRecord,
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const LENGTHS = [8_000, 32_000, 128_000, 256_000]; // tokens
const SAMPLES_PER_LENGTH = 3;
const CONTEXT_LIMIT_BASELINE = 400_000; // gpt-5 max input (approx)

const cfg: RunConfig = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  maxSteps: 30,
  maxSubCalls: 12,
  outFile: "bench/results/niah.jsonl",
  baselineContextLimitTokens: CONTEXT_LIMIT_BASELINE,
};

mkdirSync("bench/results", { recursive: true });
if (existsSync(cfg.outFile)) {
  // Clean slate each run — the file is a fresh JSONL.
  await (await import("node:fs/promises")).writeFile(cfg.outFile, "");
}

const samples = buildNIAHSweep(LENGTHS, SAMPLES_PER_LENGTH);
console.log(
  `NIAH sweep: ${samples.length} samples across ${LENGTHS.length} lengths (${SAMPLES_PER_LENGTH} each)`,
);

let totalCost = 0;
for (const s of samples) {
  // Baseline
  {
    const r = await runBaseline(
      cfg.rootModelId,
      s.question,
      s.haystack,
      s.lengthTokens,
      cfg.baselineContextLimitTokens,
    );
    const rec: RunRecord = {
      suite: "niah",
      itemId: s.id,
      condition: "baseline",
      score:
        r.error === "context_overflow"
          ? null
          : scoreNIAH(s, r.answer)
            ? 1
            : 0,
      ...r,
    };
    writeRecord(cfg, rec);
    logProgress(rec);
    totalCost += rec.costUSD;
  }
  // RLM
  {
    const r = await runRLMCondition(
      cfg,
      s.question,
      s.haystack,
      s.lengthTokens,
    );
    const rec: RunRecord = {
      suite: "niah",
      itemId: s.id,
      condition: "rlm",
      score: r.error ? 0 : scoreNIAH(s, r.answer) ? 1 : 0,
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
