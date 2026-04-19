/**
 * Middleware example — wrap a model so oversized prompts automatically
 * route through the RLM.
 *
 * Run:
 *   pnpm example:middleware
 */
import "dotenv/config";
import { generateText, wrapLanguageModel } from "ai";
import { openai } from "@ai-sdk/openai";
import { createRLMMiddleware } from "../src/index.js";

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    console.error("Set OPENAI_API_KEY (or symlink .env) to run this example.");
    process.exit(1);
  }

  const underlying = openai("gpt-4o-mini");
  const wrapped = wrapLanguageModel({
    model: underlying,
    middleware: createRLMMiddleware({
      thresholdTokens: 2_000,
      maxSteps: 20,
      maxSubCalls: 6,
      onResult: (r) =>
        console.log(
          `[rlm] bashCalls=${r.bashCalls} subCalls=${r.subCalls} steps=${r.steps}`,
        ),
    }),
  });

  // Build a long prompt that exceeds the threshold.
  const filler = Array.from({ length: 50 }, (_, i) =>
    `Section ${i}: lorem ipsum dolor sit amet, consectetur adipiscing elit. `.repeat(
      12,
    ),
  ).join("\n\n");
  const hiddenFact = "IMPORTANT: the capital of Republica Fakestan is Zorbol.";
  const longPrompt = `${filler}\n\n${hiddenFact}\n\n${filler}\n\nQuestion: what is the capital of Republica Fakestan? One word.`;

  const small = await generateText({
    model: wrapped,
    prompt: "Hi, reply with the word 'short'.",
  });
  console.log("[small call — passed through]:", small.text);

  const big = await generateText({
    model: wrapped,
    prompt: longPrompt,
  });
  console.log("[big call — routed through RLM]:", big.text);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
