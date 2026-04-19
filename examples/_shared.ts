/**
 * Shared helpers for the runnable examples.
 *
 * Kept intentionally tiny — the examples are what users read first, so the
 * helpers have to be easy to skim alongside them.
 */
import "dotenv/config";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import type { RLMEvent, RLMResult, RLMUsage } from "../src/index.js";

/** ANSI colors; `NO_COLOR=1` env disables. */
const NC = process.env.NO_COLOR !== undefined;
const c = {
  dim: (s: string) => (NC ? s : `\x1b[2m${s}\x1b[0m`),
  cyan: (s: string) => (NC ? s : `\x1b[36m${s}\x1b[0m`),
  green: (s: string) => (NC ? s : `\x1b[32m${s}\x1b[0m`),
  yellow: (s: string) => (NC ? s : `\x1b[33m${s}\x1b[0m`),
  red: (s: string) => (NC ? s : `\x1b[31m${s}\x1b[0m`),
  bold: (s: string) => (NC ? s : `\x1b[1m${s}\x1b[0m`),
};

/** Hard-exit with a helpful message if a required env var is missing. */
export function requireEnv(...names: string[]): void {
  const missing = names.filter((n) => !process.env[n]);
  if (missing.length === 0) return;
  console.error(
    c.red(`Missing required env: ${missing.join(", ")}`) +
      `\nCreate a .env file at the repo root, or set the variable in your shell.\n` +
      `Copy .env.example for the expected names.`,
  );
  process.exit(1);
}

/** Best-effort cost estimate. Prices as of 2026-04; override via env. */
const PRICING: Record<string, { input: number; output: number }> = {
  "gpt-5": { input: 1.25, output: 10.0 },
  "gpt-5-mini": { input: 0.25, output: 2.0 },
  "gpt-5-nano": { input: 0.05, output: 0.4 },
};

export function estimateCost(
  modelId: string,
  usage: RLMUsage | { inputTokens?: number; outputTokens?: number } | undefined,
): number {
  const p = PRICING[modelId];
  if (!p || !usage) return 0;
  const i = usage.inputTokens ?? 0;
  const o = usage.outputTokens ?? 0;
  return (i / 1e6) * p.input + (o / 1e6) * p.output;
}

export function fmtUsage(u: RLMUsage | undefined): string {
  if (!u) return c.dim("usage: n/a");
  return c.dim(
    `usage: ${u.inputTokens.toLocaleString()} in / ${u.outputTokens.toLocaleString()} out / ${u.totalTokens.toLocaleString()} total`,
  );
}

export function fmtCost(usd: number): string {
  return c.dim(`cost: $${usd.toFixed(4)}`);
}

/** Compact one-liner trace printer. Pass as `onEvent`. */
export function makeEventPrinter(): (e: RLMEvent) => void {
  let step = 0;
  return (e) => {
    step++;
    if (e.type === "start") {
      console.log(c.cyan(`[${step}] start`) + ` — "${e.query.slice(0, 80)}"`);
    } else if (e.type === "bash") {
      const status =
        e.result.exitCode === 0
          ? c.green(`✓ ${e.result.durationMs}ms`)
          : c.yellow(`exit=${e.result.exitCode} ${e.result.durationMs}ms`);
      const cmd = e.command.replace(/\n/g, "⏎").slice(0, 90);
      console.log(`${c.dim(`[${step}]`)} ${c.cyan("bash")}  ${status}  ${cmd}`);
    } else if (e.type === "sub-llm") {
      const preview = e.response.replace(/\s+/g, " ").slice(0, 70);
      console.log(
        `${c.dim(`[${step}]`)} ${c.cyan("llm")}   ${c.green("✓")}  → "${preview}…"`,
      );
    } else if (e.type === "final") {
      console.log(
        `${c.dim(`[${step}]`)} ${c.bold(c.green("final"))} → "${e.answer.slice(0, 80)}"`,
      );
    } else if (e.type === "error") {
      console.log(`${c.dim(`[${step}]`)} ${c.red("error")} ${e.error}`);
    }
  };
}

export function printResultSummary(
  modelId: string,
  result: RLMResult,
  wallMs: number,
): void {
  const cost = estimateCost(modelId, result.usage);
  console.log("");
  console.log(c.bold("Answer:"));
  console.log("  " + result.answer);
  console.log("");
  console.log(
    c.dim(
      `steps=${result.steps} bash=${result.bashCalls} sub=${result.subCalls} wall=${(wallMs / 1000).toFixed(1)}s`,
    ),
  );
  console.log("  " + fmtUsage(result.usage));
  console.log("  " + fmtCost(cost));
}

/** Load a bundle of files from the repo as RLM context items. */
export async function loadRepoContext(
  relPaths: string[],
): Promise<{ id: string; content: string; description: string }[]> {
  const here = dirname(fileURLToPath(import.meta.url));
  const repo = resolve(here, "..");
  const items = [];
  for (const rel of relPaths) {
    const full = join(repo, rel);
    const content = await readFile(full, "utf8");
    const id = rel.replace(/[^A-Za-z0-9]+/g, "_");
    items.push({
      id,
      content,
      description: `source file ${rel} (${content.length} bytes)`,
    });
  }
  return items;
}

export const colors = c;
