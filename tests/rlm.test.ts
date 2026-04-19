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

  it("calls leaf sub-LM via llm tool when maxDepth=0", async () => {
    let subInputSeen: string | null = null;
    const rootModel = makeScriptedModel([
      {
        toolCalls: [
          {
            toolName: "llm",
            input: {
              query: "What is 2+2? Answer only the number.",
              context: "2+2 is asked.",
            },
          },
        ],
      },
      { toolCalls: [{ toolName: "final", input: { answer: "sub said 4" } }] },
    ]);
    const subModel = makeScriptedModel([{ text: "4" }]);
    const originalDoGenerate = subModel.doGenerate.bind(subModel);
    subModel.doGenerate = async (
      params: Parameters<typeof originalDoGenerate>[0],
    ) => {
      const lastMsg = params.prompt[params.prompt.length - 1];
      if (typeof lastMsg.content === "string") subInputSeen = lastMsg.content;
      else if (Array.isArray(lastMsg.content)) {
        subInputSeen = lastMsg.content
          .map((p: { text?: string }) =>
            typeof p.text === "string" ? p.text : "",
          )
          .join("");
      }
      return originalDoGenerate(params);
    };

    const result = await runRLM(
      { model: rootModel, subModel, maxSteps: 5, maxDepth: 0 },
      { query: "ask a sub", context: "none" },
    );
    expect(result.answer).toBe("sub said 4");
    expect(result.subCalls).toBe(1);
    expect(subInputSeen).toContain("2+2");
    // Leaf path emits sub-llm event, not sub-start.
    expect(result.trace.some((e) => e.type === "sub-llm")).toBe(true);
    expect(result.trace.some((e) => e.type === "sub-start")).toBe(false);
  });

  it("recurses into a nested RLM when maxDepth>=1", async () => {
    // Root calls llm; depth 1 gets its own sandbox + bash + final.
    const rootModel = makeScriptedModel([
      {
        toolCalls: [
          {
            toolName: "llm",
            input: {
              query: "Does the snippet mention foo?",
              context: "there is a foo in here",
            },
          },
        ],
      },
      {
        toolCalls: [
          { toolName: "final", input: { answer: "root says: yes per sub" } },
        ],
      },
    ]);
    // Sub-RLM (depth 1): runs bash once, then final.
    const subCalls: string[] = [];
    const subModel = makeScriptedModel([
      {
        toolCalls: [
          { toolName: "bash", input: { command: 'cat "$RLM_CTX_CONTEXT"' } },
        ],
      },
      { toolCalls: [{ toolName: "final", input: { answer: "yes" } }] },
    ]);
    const origSubDo = subModel.doGenerate.bind(subModel);
    subModel.doGenerate = async (
      params: Parameters<typeof origSubDo>[0],
    ) => {
      subCalls.push("sub invoked");
      return origSubDo(params);
    };

    const result = await runRLM(
      { model: rootModel, subModel, maxSteps: 5, subMaxSteps: 5, maxDepth: 1 },
      { query: "root q", context: "root context" },
    );
    expect(result.answer).toBe("root says: yes per sub");
    expect(result.subCalls).toBe(1);
    // Sub-start event should fire at depth 1.
    const subStart = result.trace.find((e) => e.type === "sub-start");
    expect(subStart).toBeDefined();
    if (subStart?.type === "sub-start") expect(subStart.depth).toBe(1);
    // Sub-end event should close it.
    const subEnd = result.trace.find((e) => e.type === "sub-end");
    expect(subEnd).toBeDefined();
    // Sub-RLM runs bash at depth 1.
    const subBash = result.trace.find(
      (e) => e.type === "bash" && e.depth === 1,
    );
    expect(subBash).toBeDefined();
  });

  it("enforces maxSubCalls budget globally across nesting", async () => {
    // Root tries to call llm twice; budget is 1.
    const model = makeScriptedModel([
      {
        toolCalls: [
          {
            toolName: "llm",
            input: { query: "q1", context: "c1" },
          },
        ],
      },
      {
        toolCalls: [
          {
            toolName: "llm",
            input: { query: "q2", context: "c2" },
          },
        ],
      },
      { toolCalls: [{ toolName: "final", input: { answer: "done" } }] },
    ]);
    const subModel = makeScriptedModel([{ text: "first" }]);
    const result = await runRLM(
      { model, subModel, maxSteps: 10, maxSubCalls: 1, maxDepth: 0 },
      { query: "q", context: "c" },
    );
    expect(result.answer).toBe("done");
    expect(result.subCalls).toBe(1);
  });

  it("falls back to trailing text if final never called", async () => {
    const model = makeScriptedModel([{ text: "I think the answer is 7." }]);
    const result = await runRLM(
      { model, maxSteps: 2 },
      { query: "q", context: "c" },
    );
    expect(result.answer).toContain("7");
    // Non-fatal fallback emits a `warning` plus a synthetic `final` — not an `error`.
    const warnEvent = result.trace.find((e) => e.type === "warning");
    expect(warnEvent).toBeDefined();
    if (warnEvent?.type === "warning") {
      expect(warnEvent.message).toContain("final()");
    }
    const finalEvent = result.trace.find((e) => e.type === "final");
    expect(finalEvent).toBeDefined();
    // Should NOT have any "error" events on this happy fallback path.
    const errorEvent = result.trace.find((e) => e.type === "error");
    expect(errorEvent).toBeUndefined();
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
    await expect(
      runRLM({ model }, { query: "q", context: [] }),
    ).rejects.toThrow(/context/);
    await expect(
      runRLM({ model }, { query: "q", context: "" }),
    ).rejects.toThrow(/context/);
  });

  it("depth-1 sub-RLM's own llm calls become leaves (maxDepth=1)", async () => {
    // Root → llm → (depth 1 sub-RLM) → llm → (depth 1 leaf, since 1 !< 1).
    const rootModel = makeScriptedModel([
      {
        toolCalls: [
          {
            toolName: "llm",
            input: { query: "root-sub-q", context: "sub-context" },
          },
        ],
      },
      { toolCalls: [{ toolName: "final", input: { answer: "root-final" } }] },
    ]);
    // subModel is used for BOTH the depth-1 sub-RLM AND its leaf llm call.
    const subModel = makeScriptedModel([
      // Depth-1 sub-RLM step 1: call llm (will be a leaf since depth=1, maxDepth=1).
      {
        toolCalls: [
          {
            toolName: "llm",
            input: { query: "deeper-q", context: "deeper-ctx" },
          },
        ],
      },
      // Depth-1 sub-RLM step 2: return the leaf's answer as final.
      { toolCalls: [{ toolName: "final", input: { answer: "sub-final" } }] },
      // Leaf LLM call (plain generateText, no tools).
      { text: "leaf-answer" },
    ]);

    const result = await runRLM(
      {
        model: rootModel,
        subModel,
        maxSteps: 5,
        subMaxSteps: 5,
        maxDepth: 1,
        maxSubCalls: 10,
      },
      { query: "root q", context: "root ctx" },
    );
    expect(result.answer).toBe("root-final");
    expect(result.subCalls).toBe(2);
    // One sub-start (for the depth-1 nested RLM).
    const subStarts = result.trace.filter((e) => e.type === "sub-start");
    expect(subStarts.length).toBe(1);
    expect(subStarts[0]?.depth).toBe(1);
    // One leaf sub-llm (the depth-1 sub-RLM's own llm call became a leaf).
    const leafEvents = result.trace.filter((e) => e.type === "sub-llm");
    expect(leafEvents.length).toBe(1);
    expect(leafEvents[0]?.depth).toBe(1);
  });
});
