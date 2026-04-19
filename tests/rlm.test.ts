import { describe, expect, it } from "vitest";
import { runRLM } from "../src/rlm.js";
import { makeScriptedModel } from "./mock-model.js";

describe("runRLM (scripted model)", () => {
  it("completes when the model calls final()", async () => {
    const model = makeScriptedModel([
      { toolCalls: [{ toolName: "final", input: { answer: "42" } }] },
    ]);
    const result = await runRLM(
      { model, maxSteps: 5 },
      { query: "what is the answer?", context: "irrelevant" },
    );
    expect(result.answer).toBe("42");
    expect(result.bashCalls).toBe(0);
    expect(result.subCalls).toBe(0);
    expect(result.usage).toEqual({
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
    });
  });

  it("runs bash then final", async () => {
    const model = makeScriptedModel([
      {
        toolCalls: [
          { toolName: "bash", input: { command: 'cat "$RLM_CTX_CONTEXT"' } },
        ],
      },
      {
        toolCalls: [
          { toolName: "final", input: { answer: "found: hello world" } },
        ],
      },
    ]);
    const result = await runRLM(
      { model, maxSteps: 5 },
      { query: "read the context", context: "hello world" },
    );
    expect(result.answer).toBe("found: hello world");
    expect(result.bashCalls).toBe(1);
    const bashEvent = result.trace.find((e) => e.type === "bash");
    expect(bashEvent).toBeDefined();
    if (bashEvent?.type === "bash") {
      expect(bashEvent.result.stdout).toBe("hello world");
    }
  });

  it("calls sub-LM via llm tool", async () => {
    let subPromptSeen: string | null = null;
    const rootModel = makeScriptedModel([
      {
        toolCalls: [
          {
            toolName: "llm",
            input: { prompt: "What is 2+2? Answer only the number." },
          },
        ],
      },
      { toolCalls: [{ toolName: "final", input: { answer: "sub said 4" } }] },
    ]);
    const subModel = makeScriptedModel([{ text: "4" }]);
    const originalDoGenerate = subModel.doGenerate.bind(subModel);
    subModel.doGenerate = async (params: Parameters<typeof originalDoGenerate>[0]) => {
      const lastMsg = params.prompt[params.prompt.length - 1];
      if (typeof lastMsg.content === "string") subPromptSeen = lastMsg.content;
      else if (Array.isArray(lastMsg.content)) {
        subPromptSeen = lastMsg.content
          .map((p: { text?: string }) => (typeof p.text === "string" ? p.text : ""))
          .join("");
      }
      return originalDoGenerate(params);
    };

    const result = await runRLM(
      { model: rootModel, subModel, maxSteps: 5 },
      { query: "ask a sub", context: "none" },
    );
    expect(result.answer).toBe("sub said 4");
    expect(result.subCalls).toBe(1);
    expect(subPromptSeen).toContain("2+2");
  });

  it("enforces maxSubCalls budget", async () => {
    const model = makeScriptedModel([
      {
        toolCalls: [{ toolName: "llm", input: { prompt: "sub prompt" } }],
      },
      {
        toolCalls: [{ toolName: "llm", input: { prompt: "sub prompt 2" } }],
      },
      { toolCalls: [{ toolName: "final", input: { answer: "done" } }] },
    ]);
    const subModel = makeScriptedModel([{ text: "first" }]);
    const result = await runRLM(
      { model, subModel, maxSteps: 10, maxSubCalls: 1 },
      { query: "q", context: "c" },
    );
    expect(result.answer).toBe("done");
    expect(result.subCalls).toBe(1);
  });

  it("falls back to trailing text if final never called", async () => {
    const model = makeScriptedModel([
      { text: "I think the answer is 7." },
    ]);
    const result = await runRLM(
      { model, maxSteps: 2 },
      { query: "q", context: "c" },
    );
    expect(result.answer).toContain("7");
    const errorEvent = result.trace.find((e) => e.type === "error");
    expect(errorEvent).toBeDefined();
  });

  it("emits events through onEvent callback", async () => {
    const events: string[] = [];
    const model = makeScriptedModel([
      { toolCalls: [{ toolName: "bash", input: { command: "echo hi" } }] },
      { toolCalls: [{ toolName: "final", input: { answer: "ok" } }] },
    ]);
    await runRLM(
      { model, maxSteps: 5, onEvent: (e) => events.push(e.type) },
      { query: "q", context: "c" },
    );
    expect(events).toContain("start");
    expect(events).toContain("bash");
    expect(events).toContain("final");
  });

  it("rejects empty query", async () => {
    const model = makeScriptedModel([{ text: "n/a" }]);
    await expect(
      runRLM({ model }, { query: "   ", context: "c" }),
    ).rejects.toThrow(/query/);
  });

  it("rejects empty context", async () => {
    const model = makeScriptedModel([{ text: "n/a" }]);
    await expect(runRLM({ model }, { query: "q", context: [] })).rejects.toThrow(
      /context/,
    );
    await expect(runRLM({ model }, { query: "q", context: "" })).rejects.toThrow(
      /context/,
    );
  });
});
