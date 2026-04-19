/** 1-item OOLONG smoke test across all 3 conditions. */
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
  type RunConfig,
  type RunRecord,
} from "./runner.js";

const [item] = await fetchOolongCount32K({ n: 1 });
if (!item) throw new Error("no items");
const { query, context } = buildPrompt(item);
const labels = labelsFromQuestion(item.question);
const tokens = Math.ceil(context.length / 4);

const base: Omit<RunConfig, "maxSteps" | "maxSubCalls" | "outFile"> = {
  rootModelId: "gpt-5",
  subModelId: "gpt-5-mini",
  baselineContextLimitTokens: 400_000,
};
const cfgNoSub: RunConfig = { ...base, maxSteps: 20, maxSubCalls: 0, maxDepth: 0, outFile: "/dev/null" };
const cfgWithSub: RunConfig = { ...base, maxSteps: 20, maxSubCalls: 10, maxDepth: 1, outFile: "/dev/null" };

console.log(`Item ${item.id} task=${item.task} gold=${item.answer[0]} labels=${JSON.stringify(labels)}`);
console.log(`Context: ${tokens.toLocaleString()} tokens`);

{
  const r = await runBaseline(base.rootModelId, query, context, tokens, base.baselineContextLimitTokens);
  const rec: RunRecord = {
    suite: "oolong",
    itemId: String(item.id),
    condition: "baseline",
    score: r.error ? null : scoreOolong(item, labels, r.answer).correct ? 1 : 0,
    ...r,
  };
  logProgress(rec);
  console.log(`  answer: "${r.answer.slice(0, 120)}"`);
}
{
  const r = await runRLMCondition(cfgNoSub, query, context, tokens);
  const rec: RunRecord = {
    suite: "oolong",
    itemId: `${item.id}__no_sub`,
    condition: "rlm",
    score: r.error ? 0 : scoreOolong(item, labels, r.answer).correct ? 1 : 0,
    ...r,
  };
  logProgress(rec);
  console.log(`  answer: "${r.answer.slice(0, 120)}"`);
}
{
  const r = await runRLMCondition(cfgWithSub, query, context, tokens);
  const rec: RunRecord = {
    suite: "oolong",
    itemId: `${item.id}__with_sub`,
    condition: "rlm",
    score: r.error ? 0 : scoreOolong(item, labels, r.answer).correct ? 1 : 0,
    ...r,
  };
  logProgress(rec);
  console.log(`  answer: "${r.answer.slice(0, 120)}"`);
}
