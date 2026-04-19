import { generateText, hasToolCall, stepCountIs, tool } from "ai";
import { z } from "zod";
import { BashSandbox, estimateTokens, normaliseContext } from "./sandbox.js";
import {
  ROOT_SYSTEM_PROMPT,
  SUB_SYSTEM_PROMPT,
  buildUserPrompt,
} from "./prompts.js";
import type {
  BashResult,
  RLMEngineConfig,
  RLMEvent,
  RLMInvokeOptions,
  RLMResult,
  RLMUsage,
} from "./types.js";

const DEFAULTS = {
  maxSteps: 40,
  maxSubCalls: 20,
  bashTimeoutMs: 20_000,
  bashOutputByteCap: 8_192,
  subPromptCharCap: 400_000,
} as const;

/**
 * An RLM engine. Stateless across invocations — reuse the same engine for
 * many queries. Each `invoke()` creates and disposes a fresh bash sandbox.
 */
export class RLMEngine {
  readonly config: Required<
    Omit<RLMEngineConfig, "subModel" | "sandboxRoot" | "onEvent" | "signal">
  > &
    Pick<RLMEngineConfig, "subModel" | "sandboxRoot" | "onEvent" | "signal">;

  constructor(config: RLMEngineConfig) {
    this.config = {
      model: config.model,
      subModel: config.subModel,
      maxSteps: config.maxSteps ?? DEFAULTS.maxSteps,
      maxSubCalls: config.maxSubCalls ?? DEFAULTS.maxSubCalls,
      bashTimeoutMs: config.bashTimeoutMs ?? DEFAULTS.bashTimeoutMs,
      bashOutputByteCap: config.bashOutputByteCap ?? DEFAULTS.bashOutputByteCap,
      subPromptCharCap: config.subPromptCharCap ?? DEFAULTS.subPromptCharCap,
      sandboxRoot: config.sandboxRoot,
      onEvent: config.onEvent,
      signal: config.signal,
    };
  }

  async invoke(opts: RLMInvokeOptions): Promise<RLMResult> {
    return await runInternal(this.config, opts);
  }
}

/** Convenience one-shot: build an engine, run a query, dispose. */
export async function runRLM(
  config: RLMEngineConfig,
  opts: RLMInvokeOptions,
): Promise<RLMResult> {
  const engine = new RLMEngine(config);
  return await engine.invoke(opts);
}

async function runInternal(
  config: RLMEngine["config"],
  opts: RLMInvokeOptions,
): Promise<RLMResult> {
  if (!opts.query?.trim()) {
    throw new Error("runRLM: `query` must be a non-empty string.");
  }
  // depth is reserved for future nested-RLM recursion; events always set it so
  // UIs/tracers can display it consistently.
  const depth = 0;
  const items = normaliseContext(opts.context);
  if (items.length === 0) {
    throw new Error(
      "runRLM: `context` must contain at least one item. Pass a string, string[], or ContextItem[].",
    );
  }
  if (items.every((it) => !it.content.length)) {
    throw new Error("runRLM: all context items are empty.");
  }
  const sandbox = await BashSandbox.create({
    root: config.sandboxRoot,
    contextItems: items,
    outputByteCap: config.bashOutputByteCap,
    timeoutMs: config.bashTimeoutMs,
  });

  const trace: RLMEvent[] = [];
  const emit = (event: RLMEvent) => {
    trace.push(event);
    config.onEvent?.(event);
  };

  let subCalls = 0;
  let bashCalls = 0;
  let finalAnswer: string | null = null;
  const usage: RLMUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };

  const signal = mergeSignals(opts.signal, config.signal);

  emit({ type: "start", query: opts.query, depth });

  try {
    const tools = {
      bash: tool({
        description:
          "Execute a bash command in the sandbox. Returns combined stdout+stderr (capped). Shell state does not persist between calls; file state in $RLM_WORKDIR does.",
        inputSchema: z.object({
          command: z.string().describe("Bash command to execute."),
        }),
        execute: async ({ command }) => {
          bashCalls++;
          const result = await sandbox.execute(command, signal);
          emit({ type: "bash", command, result, depth });
          return formatBashResult(result);
        },
      }),
      llm: tool({
        description:
          "Recursively call a language model on a focused prompt. The prompt you provide is the ENTIRE input the sub-model sees — include both the instruction and the relevant snippet. Use for summarisation, extraction, comparison, or any sub-reasoning over a slice of context.",
        inputSchema: z.object({
          prompt: z
            .string()
            .describe(
              "Full prompt for the sub-LM: instruction + relevant snippet.",
            ),
        }),
        execute: async ({ prompt }) => {
          if (subCalls >= config.maxSubCalls) {
            return `ERROR: sub-LM call budget exhausted (${config.maxSubCalls}). Aggregate with what you have and call final.`;
          }
          if (prompt.length > config.subPromptCharCap) {
            return `ERROR: prompt too large (${prompt.length} chars > cap ${config.subPromptCharCap}). Split it and recurse on smaller pieces.`;
          }
          subCalls++;
          const { text, usage: subUsage } = await callSubLM(
            config,
            prompt,
            signal,
          );
          if (subUsage) accumulateUsage(usage, subUsage);
          emit({
            type: "sub-llm",
            prompt,
            response: text,
            ...(subUsage ? { usage: subUsage } : {}),
            depth,
          });
          return text;
        },
      }),
      final: tool({
        description:
          "Return the final answer and stop. Call this exactly once when you have the answer. Do not call bash or llm after this.",
        inputSchema: z.object({
          answer: z
            .string()
            .describe("The final natural-language answer to the user query."),
        }),
        execute: async ({ answer }) => {
          finalAnswer = answer;
          emit({ type: "final", answer, depth });
          return "ok";
        },
      }),
    };

    const result = await generateText({
      model: config.model,
      system: ROOT_SYSTEM_PROMPT,
      prompt: buildUserPrompt(
        opts.query,
        sandbox.describeContext(),
        opts.additionalInstructions,
      ),
      tools,
      stopWhen: [stepCountIs(config.maxSteps), hasToolCall("final")],
      abortSignal: signal,
    });

    const rootUsage = extractUsage(result.usage);
    if (rootUsage) accumulateUsage(usage, rootUsage);

    if (finalAnswer === null) {
      // Model exited without calling final — fall back to its text.
      finalAnswer = result.text?.trim() || "(no answer produced)";
      emit({
        type: "error",
        error: "RLM exited without final() — used trailing text as fallback.",
        depth,
      });
    }

    return {
      answer: finalAnswer,
      steps: result.steps?.length ?? 0,
      subCalls,
      bashCalls,
      usage,
      trace,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    emit({ type: "error", error: message, depth });
    throw err;
  } finally {
    await sandbox.dispose();
  }
}

function formatBashResult(result: BashResult): string {
  const parts: string[] = [];
  parts.push(`exit=${result.exitCode} duration=${result.durationMs}ms`);
  if (result.timedOut) parts.push("TIMED_OUT");
  if (result.truncated) parts.push("TRUNCATED");
  const meta = parts.join(" ");
  const body = [
    result.stdout ? `--- stdout ---\n${result.stdout}` : "",
    result.stderr ? `--- stderr ---\n${result.stderr}` : "",
  ]
    .filter(Boolean)
    .join("\n");
  return `[${meta}]\n${body || "(no output)"}`;
}

async function callSubLM(
  config: RLMEngine["config"],
  prompt: string,
  signal?: AbortSignal,
): Promise<{ text: string; usage?: RLMUsage }> {
  const model = config.subModel ?? config.model;
  const res = await generateText({
    model,
    system: SUB_SYSTEM_PROMPT,
    prompt,
    abortSignal: signal,
  });
  const u = extractUsage(res.usage);
  return u ? { text: res.text, usage: u } : { text: res.text };
}

/** Convert the AI SDK's usage shape to our flat RLMUsage. Returns undefined
 *  if no token counts are reported. */
function extractUsage(u: unknown): RLMUsage | undefined {
  if (!u || typeof u !== "object") return undefined;
  const rec = u as Record<string, unknown>;
  const input =
    typeof rec.inputTokens === "number" ? rec.inputTokens : undefined;
  const output =
    typeof rec.outputTokens === "number" ? rec.outputTokens : undefined;
  const total =
    typeof rec.totalTokens === "number"
      ? rec.totalTokens
      : input !== undefined && output !== undefined
        ? input + output
        : undefined;
  if (input === undefined && output === undefined && total === undefined) {
    return undefined;
  }
  return {
    inputTokens: input ?? 0,
    outputTokens: output ?? 0,
    totalTokens: total ?? (input ?? 0) + (output ?? 0),
  };
}

function accumulateUsage(into: RLMUsage, add: RLMUsage): void {
  into.inputTokens += add.inputTokens;
  into.outputTokens += add.outputTokens;
  into.totalTokens += add.totalTokens;
}

function mergeSignals(
  a: AbortSignal | undefined,
  b: AbortSignal | undefined,
): AbortSignal | undefined {
  if (!a) return b;
  if (!b) return a;
  // Node ≥20 (our `engines` floor) ships AbortSignal.any — no fallback needed.
  return AbortSignal.any([a, b]);
}

// Re-export token helper for consumers wanting to decide themselves when to
// route a call through the RLM.
export { estimateTokens };
