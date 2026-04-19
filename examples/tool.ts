/**
 * Tool example — multi-turn agent with an `rlm` tool it can call at will.
 *
 * We give the agent a large document (this repo's source) and ask it a
 * compound question. The agent decides for itself when to call the rlm
 * tool vs answer from memory, and can call it multiple times across the
 * conversation.
 *
 *   pnpm example:tool
 */
import { generateText, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { rlmTool } from "../src/index.js";
import { colors, estimateCost, fmtCost, fmtUsage, requireEnv } from "./_shared.js";

requireEnv("OPENAI_API_KEY");

const AGENT_MODEL = "gpt-5";
const RLM_ROOT = "gpt-5";
const RLM_SUB = "gpt-5-mini";

const here = dirname(fileURLToPath(import.meta.url));
const repo = async (rel: string) =>
  `// ===== ${rel} =====\n` +
  (await readFile(join(here, "..", rel), "utf8"));
const document = [
  await repo("src/rlm.ts"),
  await repo("src/sandbox.ts"),
  await repo("src/types.ts"),
  await repo("src/prompts.ts"),
].join("\n\n");

const start = Date.now();
const result = await generateText({
  model: openai(AGENT_MODEL),
  system: [
    "You are a senior code reviewer. You have a tool called `rlm` that can",
    "answer questions against a large document (the full text of which you",
    "should NOT try to read yourself — pass it as `context` to the tool).",
    "Use the tool as many times as you need. When you're done, summarise.",
  ].join(" "),
  prompt: [
    "I'm giving you the source of a TypeScript package. Please answer:",
    "",
    "  1) What does the persistent REPL sandbox use as the 'end of command' signal?",
    "  2) What is the default max number of recursive sub-LLM calls per invocation?",
    "  3) Name one thing the root-LM system prompt explicitly tells the model NOT to do.",
    "",
    "Think step-by-step. Call the `rlm` tool (once or many times) to find the answers.",
    "When finished, present 1/2/3 as a numbered list.",
  ].join("\n"),
  tools: {
    rlm: rlmTool({
      model: openai(RLM_ROOT),
      subModel: openai(RLM_SUB),
      maxSteps: 25,
      maxSubCalls: 6,
    }),
  },
  stopWhen: stepCountIs(8),
  // Pass the document as a tool parameter via the agent; we seed it into a
  // lightweight wrapper by supplying it through system context instead.
  prepareStep: async ({ stepNumber }) => {
    if (stepNumber === 0) {
      return {
        // Give the agent the document via the first user turn.
        messages: [
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const,
                text:
                  `<document tag="rlm-ai-sdk source">\n${document}\n</document>`,
              },
            ],
          },
        ],
      };
    }
    return {};
  },
});

console.log(colors.bold("=== Agent final ==="));
console.log(result.text);

console.log(
  "\n" +
    colors.dim(
      `agent steps=${result.steps?.length ?? 0} tool calls=${result.steps
        ?.flatMap((s) => s.toolCalls ?? [])
        .length ?? 0}`,
    ),
);
if (result.usage) {
  console.log(
    "  " +
      fmtUsage({
        inputTokens: result.usage.inputTokens ?? 0,
        outputTokens: result.usage.outputTokens ?? 0,
        totalTokens:
          (result.usage.inputTokens ?? 0) + (result.usage.outputTokens ?? 0),
      }),
  );
  console.log(
    "  " +
      fmtCost(
        estimateCost(AGENT_MODEL, {
          inputTokens: result.usage.inputTokens ?? 0,
          outputTokens: result.usage.outputTokens ?? 0,
        }),
      ),
  );
}
console.log("  " + colors.dim(`wall=${((Date.now() - start) / 1000).toFixed(1)}s`));
