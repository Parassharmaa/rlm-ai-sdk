import { tool } from "ai";
import { z } from "zod";
import { runRLM } from "./rlm.js";
import type { RLMEngineConfig } from "./types.js";

/**
 * Exposes the RLM as a standard AI SDK tool. Drop it into any
 * `generateText({ tools: { ... } })` call to give a model the ability to
 * spin up a recursive sub-model over a bounded context.
 *
 * The outer agent provides a `query` and a `context` (string or string[]);
 * the RLM runs its bash-sandboxed exploration and returns just the answer
 * string to the agent.
 */
export function rlmTool(config: RLMEngineConfig) {
  return tool({
    description:
      "Recursively explore a large context to answer a question. The context is stored in a bash sandbox and explored programmatically. Use this when you need to answer a question over a body of text that would be too large to quote in full.",
    inputSchema: z.object({
      query: z
        .string()
        .describe("The natural-language question to answer over the context."),
      context: z
        .union([z.string(), z.array(z.string())])
        .describe(
          "The context to explore. A single string or an array of chunks.",
        ),
      additionalInstructions: z
        .string()
        .optional()
        .describe(
          "Optional extra instructions or hints for the RLM root model.",
        ),
    }),
    execute: async ({ query, context, additionalInstructions }) => {
      const invokeOpts: Parameters<typeof runRLM>[1] = { query, context };
      if (additionalInstructions !== undefined) {
        invokeOpts.additionalInstructions = additionalInstructions;
      }
      const result = await runRLM(config, invokeOpts);
      return {
        answer: result.answer,
        steps: result.steps,
        subCalls: result.subCalls,
        bashCalls: result.bashCalls,
      };
    },
  });
}
