import { MockLanguageModelV3 } from "ai/test";

export type MockModel = InstanceType<typeof MockLanguageModelV3>;

/**
 * Build a scripted MockLanguageModelV3 that plays back a sequence of
 * responses. Each response is either a plain text reply or a list of
 * tool-call directives.
 *
 * A scripted response shape:
 *   { text: "..." }                                             — final text
 *   { toolCalls: [{ toolName, input: { ... } }, ...] }          — one step
 *   { toolCalls: [...], text: "trailing" }                      — both
 *
 * After the script runs out, the model returns empty text / stop.
 */
export type ScriptedStep =
  | { text: string }
  | {
      toolCalls: Array<{
        toolName: string;
        input: Record<string, unknown>;
        toolCallId?: string;
      }>;
      text?: string;
    };

export function makeScriptedModel(script: ScriptedStep[]): MockModel {
  let step = 0;
  let streamStep = 0;
  return new MockLanguageModelV3({
    // doStream for tests that exercise streamText against this model.
    // Emits the current scripted step as text-start / text-delta / text-end
    // / finish. Tool calls aren't streamed here — this is for pass-through
    // streaming tests where the scripted step is plain text.
    doStream: async () => {
      const current = script[streamStep] ?? { text: "" };
      streamStep++;
      const textContent =
        "text" in current && current.text ? current.text : "";
      const id = `stream_${streamStep}`;
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue({ type: "text-start", id });
          if (textContent) {
            controller.enqueue({ type: "text-delta", id, delta: textContent });
          }
          controller.enqueue({ type: "text-end", id });
          controller.enqueue({
            type: "finish",
            finishReason: { unified: "stop", raw: undefined },
            usage: {
              inputTokens: {
                total: 0,
                noCache: 0,
                cacheRead: 0,
                cacheWrite: 0,
              },
              outputTokens: { total: 0, text: 0, reasoning: 0 },
            },
          });
          controller.close();
        },
      });
      return { stream };
    },
    doGenerate: async () => {
      const current = script[step] ?? { text: "" };
      step++;
      const content: Array<{ type: "text"; text: string } | { type: "tool-call"; toolCallId: string; toolName: string; input: string }> = [];
      if ("toolCalls" in current && current.toolCalls) {
        for (const tc of current.toolCalls) {
          content.push({
            type: "tool-call",
            toolCallId: tc.toolCallId ?? `call_${step}_${tc.toolName}`,
            toolName: tc.toolName,
            input: JSON.stringify(tc.input),
          });
        }
      }
      if ("text" in current && current.text) {
        content.push({ type: "text", text: current.text });
      }
      const hasToolCalls = content.some((c) => c.type === "tool-call");
      return {
        content,
        finishReason: {
          unified: hasToolCalls ? ("tool-calls" as const) : ("stop" as const),
          raw: undefined,
        },
        usage: {
          inputTokens: {
            total: 0,
            noCache: 0,
            cacheRead: 0,
            cacheWrite: 0,
          },
          outputTokens: { total: 0, text: 0, reasoning: 0 },
        },
        warnings: [],
      };
    },
  });
}
