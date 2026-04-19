import { describe, expect, it } from "vitest";
import { generateText, streamText, wrapLanguageModel } from "ai";
import { createRLMMiddleware } from "../src/middleware.js";
import { makeScriptedModel } from "./mock-model.js";

describe("createRLMMiddleware", () => {
  it("passes small prompts through unchanged", async () => {
    const underlying = makeScriptedModel([{ text: "plain response" }]);
    const wrapped = wrapLanguageModel({
      model: underlying,
      middleware: createRLMMiddleware({ thresholdTokens: 1_000_000 }),
    });
    const result = await generateText({
      model: wrapped,
      prompt: "tiny prompt",
    });
    expect(result.text).toBe("plain response");
  });

  it("diverts large prompts through RLM and returns the RLM answer", async () => {
    // Root model: calls final immediately with "rlm-answer".
    const rootModel = makeScriptedModel([
      {
        toolCalls: [
          { toolName: "final", input: { answer: "rlm-answer" } },
        ],
      },
    ]);
    const wrapped = wrapLanguageModel({
      model: rootModel,
      middleware: createRLMMiddleware({
        thresholdTokens: 0, // always divert
      }),
    });
    const huge = "x".repeat(200_000);
    const result = await generateText({
      model: wrapped,
      prompt: `Summarise: ${huge}`,
    });
    expect(result.text).toBe("rlm-answer");
    expect(
      (result.providerMetadata as { "rlm-ai-sdk"?: { steps?: number } } | undefined)?.[
        "rlm-ai-sdk"
      ],
    ).toBeDefined();
  });

  it("passes through streamText unchanged for small prompts", async () => {
    const underlying = makeScriptedModel([{ text: "streamed-passthrough" }]);
    const wrapped = wrapLanguageModel({
      model: underlying,
      middleware: createRLMMiddleware({ thresholdTokens: 1_000_000 }),
    });
    const stream = streamText({
      model: wrapped,
      prompt: "short prompt",
    });
    let accumulated = "";
    for await (const delta of stream.textStream) accumulated += delta;
    expect(accumulated).toBe("streamed-passthrough");
  });

  it("streams diverted responses as a synthetic text stream", async () => {
    const rootModel = makeScriptedModel([
      { toolCalls: [{ toolName: "final", input: { answer: "streamed-answer" } }] },
    ]);
    const wrapped = wrapLanguageModel({
      model: rootModel,
      middleware: createRLMMiddleware({ thresholdTokens: 0 }),
    });
    const stream = streamText({
      model: wrapped,
      prompt: `Summarise: ${"x".repeat(200_000)}`,
    });
    let accumulated = "";
    for await (const delta of stream.textStream) accumulated += delta;
    expect(accumulated).toBe("streamed-answer");
  });

  it("respects shouldRoute predicate", async () => {
    let shouldRouteCalls = 0;
    const underlying = makeScriptedModel([
      { text: "never-called-if-diverted" },
      { text: "never-called-if-diverted" },
    ]);
    const wrapped = wrapLanguageModel({
      model: underlying,
      middleware: createRLMMiddleware({
        shouldRoute: () => {
          shouldRouteCalls++;
          return false; // always pass through
        },
      }),
    });
    const result = await generateText({
      model: wrapped,
      prompt: "whatever",
    });
    expect(result.text).toBe("never-called-if-diverted");
    expect(shouldRouteCalls).toBe(1);
  });
});
