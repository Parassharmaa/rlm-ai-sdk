/**
 * Aggregate JSONL results into a summary table printed as markdown.
 *
 *   pnpm tsx bench/summarize.ts
 */
import { readFileSync, existsSync } from "node:fs";
import type { RunRecord } from "./runner.js";

function loadJsonl(path: string): RunRecord[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as RunRecord);
}

function pct(n: number, d: number): string {
  if (d === 0) return "—";
  return `${((n / d) * 100).toFixed(1)}%`;
}

function bucketize(records: RunRecord[]): Map<string, RunRecord[]> {
  const map = new Map<string, RunRecord[]>();
  for (const r of records) {
    const kTokens = Math.round(r.contextTokens / 1000);
    // NIAH bucket by configured length (snap to nearest power-of-two-ish).
    let bucket = `${kTokens}K`;
    if (r.suite === "niah") {
      const target = [8, 32, 128, 256].reduce((best, t) =>
        Math.abs(kTokens - t) < Math.abs(kTokens - best) ? t : best,
      );
      bucket = `${target}K`;
    }
    const key = `${r.suite}/${bucket}/${r.condition}`;
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(r);
  }
  return map;
}

function summariseNIAH(records: RunRecord[]): string {
  const out: string[] = [];
  const niah = records.filter((r) => r.suite === "niah");
  if (niah.length === 0) return "";
  const b = bucketize(niah);
  const lengths = ["8K", "32K", "128K", "256K"];
  out.push("## S-NIAH (single needle in a haystack)\n");
  out.push(
    "| Length | Baseline (gpt-5) | RLM (gpt-5 + gpt-5-mini) | Base $ | RLM $ | Base time | RLM time |",
  );
  out.push(
    "|--------|------------------|---------------------------|--------|-------|-----------|----------|",
  );
  let totalBaseCost = 0;
  let totalRlmCost = 0;
  for (const len of lengths) {
    const base = b.get(`niah/${len}/baseline`) ?? [];
    const rlm = b.get(`niah/${len}/rlm`) ?? [];
    const basePass = base.filter((r) => r.score === 1).length;
    const baseRan = base.filter((r) => r.score !== null).length;
    const baseSkip = base.filter((r) => r.score === null).length;
    const rlmPass = rlm.filter((r) => r.score === 1).length;
    const baseCost = base.reduce((s, r) => s + r.costUSD, 0);
    const rlmCost = rlm.reduce((s, r) => s + r.costUSD, 0);
    totalBaseCost += baseCost;
    totalRlmCost += rlmCost;
    const baseCell =
      baseRan === 0
        ? `overflow (n=${baseSkip})`
        : `${pct(basePass, baseRan)} (${basePass}/${baseRan}${baseSkip ? `, ${baseSkip} skip` : ""})`;
    const rlmCell =
      rlm.length === 0
        ? "—"
        : `${pct(rlmPass, rlm.length)} (${rlmPass}/${rlm.length})`;
    const avg = (rs: RunRecord[]) =>
      rs.length ? (rs.reduce((s, r) => s + r.elapsedMs, 0) / rs.length / 1000).toFixed(1) + "s" : "—";
    out.push(
      `| ${len} | ${baseCell} | ${rlmCell} | $${baseCost.toFixed(3)} | $${rlmCost.toFixed(3)} | ${avg(base)} | ${avg(rlm)} |`,
    );
  }
  out.push(
    `\n**Totals:** baseline $${totalBaseCost.toFixed(2)}, RLM $${totalRlmCost.toFixed(2)}\n`,
  );
  return out.join("\n");
}

function summariseCodeQA(records: RunRecord[]): string {
  const out: string[] = [];
  const code = records.filter((r) => r.suite === "codeqa");
  if (code.length === 0) return "";
  out.push("\n## LongBench-v2 CodeQA subset\n");
  const base = code.filter((r) => r.condition === "baseline");
  const rlm = code.filter((r) => r.condition === "rlm");
  const basePass = base.filter((r) => r.score === 1).length;
  const baseRan = base.filter((r) => r.score !== null).length;
  const rlmPass = rlm.filter((r) => r.score === 1).length;
  const baseCost = base.reduce((s, r) => s + r.costUSD, 0);
  const rlmCost = rlm.reduce((s, r) => s + r.costUSD, 0);
  out.push(`- Baseline (gpt-5 direct): **${pct(basePass, baseRan)}** (${basePass}/${baseRan}), cost $${baseCost.toFixed(2)}`);
  out.push(`- RLM (gpt-5 + gpt-5-mini): **${pct(rlmPass, rlm.length)}** (${rlmPass}/${rlm.length}), cost $${rlmCost.toFixed(2)}`);
  out.push("\n### Per-item");
  out.push("| Item | Tokens | Base | RLM | RLM steps/bash/sub |");
  out.push("|------|--------|------|-----|--------------------|");
  const byId: Record<string, { baseline?: RunRecord; rlm?: RunRecord }> = {};
  for (const r of code) {
    byId[r.itemId] = byId[r.itemId] ?? {};
    byId[r.itemId]![r.condition] = r;
  }
  for (const id of Object.keys(byId).sort()) {
    const { baseline, rlm } = byId[id]!;
    const tk = ((baseline ?? rlm)!.contextTokens / 1000).toFixed(0);
    const bCell = baseline
      ? baseline.score === null
        ? "SKIP"
        : baseline.score === 1
          ? "✅"
          : "❌"
      : "—";
    const rCell = rlm ? (rlm.score === 1 ? "✅" : "❌") : "—";
    const steps = rlm ? `${rlm.steps}/${rlm.bashCalls}/${rlm.subCalls}` : "—";
    out.push(`| ${id.slice(0, 8)} | ${tk}K | ${bCell} | ${rCell} | ${steps} |`);
  }
  return out.join("\n");
}

function summariseOolong(records: RunRecord[]): string {
  const out: string[] = [];
  const oolong = records.filter((r) => r.suite === "oolong");
  if (oolong.length === 0) return "";
  out.push("\n## OOLONG counting @ 32K (N=10)\n");
  out.push(
    "Task: MOST_FREQ / LEAST_FREQ over 80 binary-labelled items. Aggregation task — requires classifying every item and counting. Paper comparable: OOLONG-Pairs (quadratic aggregation).\n",
  );
  const base = oolong.filter((r) => r.condition === "baseline");
  // RLM records use suffixed itemIds to distinguish ablations.
  const rlmNoSub = oolong.filter((r) => r.condition === "rlm" && r.itemId.endsWith("__no_sub"));
  const rlmWithSub = oolong.filter(
    (r) => r.condition === "rlm" && r.itemId.endsWith("__with_sub"),
  );

  const stats = (rs: RunRecord[]) => {
    const ran = rs.filter((r) => r.score !== null);
    const pass = rs.filter((r) => r.score === 1).length;
    const cost = rs.reduce((s, r) => s + r.costUSD, 0);
    const elapsed = rs.reduce((s, r) => s + r.elapsedMs, 0);
    return {
      pct: ran.length > 0 ? pct(pass, ran.length) : "—",
      fraction: `${pass}/${ran.length}`,
      cost: cost.toFixed(2),
      avgS: rs.length ? (elapsed / rs.length / 1000).toFixed(1) : "—",
    };
  };
  const b = stats(base);
  const n = stats(rlmNoSub);
  const w = stats(rlmWithSub);
  out.push(
    "| Condition | Accuracy | Total cost | Avg wall time |",
  );
  out.push("|---|---|---|---|");
  out.push(`| Baseline (gpt-5 direct) | **${b.pct}** (${b.fraction}) | $${b.cost} | ${b.avgS}s |`);
  out.push(`| RLM no-sub (bash only, ablation) | **${n.pct}** (${n.fraction}) | $${n.cost} | ${n.avgS}s |`);
  out.push(`| RLM w/ leaf sub-calls (maxDepth=0, maxSubCalls=10) | **${w.pct}** (${w.fraction}) | $${w.cost} | ${w.avgS}s |`);
  return out.join("\n");
}

const all: RunRecord[] = [
  ...loadJsonl("bench/results/niah.jsonl"),
  ...loadJsonl("bench/results/codeqa.jsonl"),
  ...loadJsonl("bench/results/oolong.jsonl"),
];

if (all.length === 0) {
  console.log(
    "No results yet. Run bench/run-niah.ts, bench/run-codeqa.ts, or bench/run-oolong.ts first.",
  );
  process.exit(0);
}

console.log("# RLM benchmarks — gpt-5 root / gpt-5-mini sub\n");
console.log(
  `Generated: ${new Date().toISOString().replace("T", " ").slice(0, 19)}Z`,
);
console.log(summariseNIAH(all));
console.log(summariseCodeQA(all));
console.log(summariseOolong(all));
