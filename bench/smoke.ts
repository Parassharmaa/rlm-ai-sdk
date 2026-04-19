/** Micro smoke test — 1 NIAH sample at 4K tokens through both conditions. */
import { generateNIAH, scoreNIAH } from "./niah.js";
import {
  logProgress,
  runBaseline,
  runRLMCondition,
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const cfg: RunConfig = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  maxSteps: 15,
  maxSubCalls: 5,
  outFile: "bench/results/smoke.jsonl",
  baselineContextLimitTokens: 400_000,
};

const s = generateNIAH(4_000, 42);
console.log(`sample: ${s.id} tokens=${s.lengthTokens} needle=${s.magicNumber}`);

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
      r.error === "context_overflow" ? null : scoreNIAH(s, r.answer) ? 1 : 0,
    ...r,
  };
  logProgress(rec);
  console.log(`  answer: "${r.answer.slice(0, 200)}"`);
}
{
  const r = await runRLMCondition(cfg, s.question, s.haystack, s.lengthTokens);
  const rec: RunRecord = {
    suite: "niah",
    itemId: s.id,
    condition: "rlm",
    score: r.error ? 0 : scoreNIAH(s, r.answer) ? 1 : 0,
    ...r,
  };
  logProgress(rec);
  console.log(`  answer: "${r.answer.slice(0, 200)}"`);
}
