/**
 * Middleware example — wrap any AI SDK model so oversized prompts auto-route
 * through the RLM. Demonstrates both `generateText` and `streamText` paths.
 *
 *   pnpm example:middleware
 */
import { generateText, streamText, wrapLanguageModel } from "ai";
import { openai } from "@ai-sdk/openai";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRLMMiddleware } from "../src/index.js";
import {
  colors,
  estimateCost,
  fmtCost,
  fmtUsage,
  requireEnv,
} from "./_shared.js";

requireEnv("OPENAI_API_KEY");

const MODEL = "gpt-5-mini";
const THRESHOLD = 4_000; // tokens; below → pass-through, above → RLM

const wrapped = wrapLanguageModel({
  model: openai(MODEL),
  middleware: createRLMMiddleware({
    thresholdTokens: THRESHOLD,
    model: openai("gpt-5"), // bigger model drives the RLM loop
    subModel: openai("gpt-5-mini"),
    maxSteps: 20,
    maxSubCalls: 6,
    onResult: (r) =>
      console.log(
        colors.dim(
          `  [rlm] steps=${r.steps} bash=${r.bashCalls} sub=${r.subCalls} usage=${r.usage.totalTokens} tok`,
        ),
      ),
  }),
});

// Build a large, real-ish document by concatenating the repo's own source.
const here = dirname(fileURLToPath(import.meta.url));
const files = ["src/rlm.ts", "src/sandbox.ts", "src/middleware.ts"];
const docs = await Promise.all(
  files.map(async (f) =>
    `// ===== ${f} =====\n` + (await readFile(join(here, "..", f), "utf8")),
  ),
);
const longContext = docs.join("\n\n");
const longTokens = Math.ceil(longContext.length / 4);

console.log(
  colors.bold("=== 1. Small prompt — passes through to underlying model ===\n"),
);
const smallStart = Date.now();
const small = await generateText({
  model: wrapped,
  prompt: "In one short sentence, what is a recursive language model?",
});
console.log("Answer:", small.text.trim());
console.log(
  "  " +
    fmtUsage({
      inputTokens: small.usage?.inputTokens ?? 0,
      outputTokens: small.usage?.outputTokens ?? 0,
      totalTokens:
        (small.usage?.inputTokens ?? 0) + (small.usage?.outputTokens ?? 0),
    }),
);
console.log(
  "  " +
    fmtCost(
      estimateCost(MODEL, {
        inputTokens: small.usage?.inputTokens ?? 0,
        outputTokens: small.usage?.outputTokens ?? 0,
      }),
    ),
);
console.log("  " + colors.dim(`wall=${Date.now() - smallStart}ms`));

console.log(
  "\n" +
    colors.bold(
      `=== 2. Large prompt (~${longTokens.toLocaleString()} tokens) — diverted via generateText ===\n`,
    ),
);
const bigStart = Date.now();
const big = await generateText({
  model: wrapped,
  prompt:
    longContext +
    "\n\nQuestion: Which function in src/sandbox.ts handles the done-marker " +
    "scanning, and what does it do when the output cap is reached before the " +
    "marker arrives? Answer in one sentence.",
});
console.log("Answer:", big.text.trim());
console.log("  " + colors.dim(`wall=${((Date.now() - bigStart) / 1000).toFixed(1)}s`));

console.log(
  "\n" +
    colors.bold(
      `=== 3. Same large prompt — diverted via streamText (synthetic stream) ===\n`,
    ),
);
process.stdout.write("Stream: ");
const stream = streamText({
  model: wrapped,
  prompt:
    longContext +
    "\n\nQuestion: What is the default value of maxSteps in the RLM engine config? Reply with just the number.",
});
for await (const delta of stream.textStream) process.stdout.write(delta);
process.stdout.write("\n");
