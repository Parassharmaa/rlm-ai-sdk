/**
 * A/B runner for RLM vs baseline. Emits one JSONL line per (item, condition).
 *
 * Cost model: takes per-model $/M input, $/M output tokens. Tracks tokens
 * and dollars per run. Hard step/call caps prevent runaway RLM loops.
 */
import "dotenv/config";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";
import { appendFileSync } from "node:fs";
import { runRLM } from "../src/index.js";
import type { LanguageModel } from "ai";

export interface Pricing {
  inputPerM: number; // $ per 1M input tokens
  outputPerM: number; // $ per 1M output tokens
}

export const PRICES: Record<string, Pricing> = {
  // OpenAI list prices as of 2026-04. If GPT-5 pricing changes, override via env.
  "gpt-5": { inputPerM: 1.25, outputPerM: 10.0 },
  "gpt-5-mini": { inputPerM: 0.25, outputPerM: 2.0 },
};

export interface RunConfig {
  rootModelId: string;
  subModelId: string;
  maxSteps: number;
  maxSubCalls: number;
  maxDepth?: number;
  outFile: string;
  // If baseline context exceeds this, skip baseline and mark as "context_overflow".
  baselineContextLimitTokens: number;
}

export interface RunRecord {
  suite: "niah" | "codeqa" | "oolong";
  itemId: string;
  condition: "baseline" | "rlm";
  contextTokens: number;
  answer: string;
  score: 0 | 1 | null; // null = not run (overflow / error)
  error: string | null;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  costUSD: number;
  elapsedMs: number;
  // RLM extras
  steps?: number;
  bashCalls?: number;
  subCalls?: number;
}

function cost(modelId: string, input: number, output: number): number {
  const p = PRICES[modelId];
  if (!p) return 0;
  return (input / 1e6) * p.inputPerM + (output / 1e6) * p.outputPerM;
}

export async function runBaseline(
  rootModelId: string,
  query: string,
  context: string,
  contextTokens: number,
  limitTokens: number,
  signal?: AbortSignal,
): Promise<Omit<RunRecord, "suite" | "itemId" | "condition" | "score">> {
  if (contextTokens > limitTokens) {
    return {
      contextTokens,
      answer: "",
      error: "context_overflow",
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      costUSD: 0,
      elapsedMs: 0,
    };
  }
  const t0 = Date.now();
  try {
    const res = await generateText({
      model: openai(rootModelId as never) as LanguageModel,
      system:
        "You are a careful analyst. Answer the user's question using only the provided context. Be concise.",
      prompt: `# Context\n${context}\n\n# Question\n${query}`,
      abortSignal: signal,
    });
    const input = res.usage?.inputTokens ?? 0;
    const output = res.usage?.outputTokens ?? 0;
    return {
      contextTokens,
      answer: res.text,
      error: null,
      inputTokens: input,
      outputTokens: output,
      totalTokens: input + output,
      costUSD: cost(rootModelId, input, output),
      elapsedMs: Date.now() - t0,
    };
  } catch (e) {
    return {
      contextTokens,
      answer: "",
      error: (e as Error).message.slice(0, 300),
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      costUSD: 0,
      elapsedMs: Date.now() - t0,
    };
  }
}

export async function runRLMCondition(
  cfg: RunConfig,
  query: string,
  context: string | string[],
  contextTokens: number,
  signal?: AbortSignal,
): Promise<Omit<RunRecord, "suite" | "itemId" | "condition" | "score">> {
  const t0 = Date.now();
  try {
    const res = await runRLM(
      {
        model: openai(cfg.rootModelId as never) as LanguageModel,
        subModel: openai(cfg.subModelId as never) as LanguageModel,
        maxSteps: cfg.maxSteps,
        maxSubCalls: cfg.maxSubCalls,
        ...(cfg.maxDepth !== undefined ? { maxDepth: cfg.maxDepth } : {}),
      },
      { query, context, signal },
    );
    // Approximate cost: we don't split tokens by root vs sub here; price all
    // at the root model's rate (upper bound) — a simpler alternative would
    // be to weight by sub-call count, but we don't track per-call usage.
    const input = res.usage.inputTokens;
    const output = res.usage.outputTokens;
    return {
      contextTokens,
      answer: res.answer,
      error: null,
      inputTokens: input,
      outputTokens: output,
      totalTokens: res.usage.totalTokens,
      costUSD: cost(cfg.rootModelId, input, output),
      elapsedMs: Date.now() - t0,
      steps: res.steps,
      bashCalls: res.bashCalls,
      subCalls: res.subCalls,
    };
  } catch (e) {
    return {
      contextTokens,
      answer: "",
      error: (e as Error).message.slice(0, 300),
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      costUSD: 0,
      elapsedMs: Date.now() - t0,
    };
  }
}

export function writeRecord(cfg: RunConfig, rec: RunRecord): void {
  appendFileSync(cfg.outFile, JSON.stringify(rec) + "\n");
}

/** Print a tight one-liner for progress. */
export function logProgress(rec: RunRecord): void {
  const score =
    rec.score === null ? "SKIP" : rec.score === 1 ? "PASS" : "FAIL";
  const extra = rec.condition === "rlm"
    ? ` steps=${rec.steps} bash=${rec.bashCalls} sub=${rec.subCalls}`
    : "";
  console.log(
    `[${rec.suite}] ${rec.itemId} ${rec.condition.padEnd(8)} ${score} ${(rec.contextTokens / 1000).toFixed(1)}K $${rec.costUSD.toFixed(3)} ${(rec.elapsedMs / 1000).toFixed(1)}s${extra}${
      rec.error ? ` err=${rec.error.slice(0, 60)}` : ""
    }`,
  );
}
