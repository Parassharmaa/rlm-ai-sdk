/**
 * Basic RLM example — code understanding over the rlm-ai-sdk repo itself.
 *
 * The "context" is this very package's source code. We ask the RLM a
 * question that requires navigating multiple files, then print the full
 * trace so you can see exactly how the root LM drove the persistent bash
 * REPL to answer.
 *
 *   pnpm example:basic
 */
import {
  loadRepoContext,
  makeEventPrinter,
  printResultSummary,
  requireEnv,
} from "./_shared.js";
import { openai } from "@ai-sdk/openai";
import { runRLM } from "../src/index.js";

requireEnv("OPENAI_API_KEY");

const ROOT = "gpt-5";
const SUB = "gpt-5-mini";

const context = await loadRepoContext([
  "src/rlm.ts",
  "src/sandbox.ts",
  "src/middleware.ts",
  "src/tool.ts",
  "src/types.ts",
  "src/prompts.ts",
  "README.md",
]);

const totalChars = context.reduce((n, c) => n + c.content.length, 0);
console.log(
  `Context: ${context.length} files, ${totalChars.toLocaleString()} chars (~${Math.ceil(totalChars / 4 / 1000)}K tokens).\n`,
);

const start = Date.now();
const result = await runRLM(
  {
    model: openai(ROOT),
    subModel: openai(SUB),
    maxSteps: 30,
    maxSubCalls: 8,
    onEvent: makeEventPrinter(),
  },
  {
    query:
      "In this codebase, explain concretely how the RLM prevents output from exceeding its configured byte cap. Which file implements it, and what happens if the cap is hit while the done-marker is still arriving? Answer in 2-3 sentences citing the relevant function name.",
    context,
  },
);

printResultSummary(ROOT, result, Date.now() - start);
