import { generateText, hasToolCall, stepCountIs, tool } from "ai";
import { z } from "zod";
import { BashSandbox, estimateTokens, normaliseContext } from "./sandbox.js";
import {
  ROOT_SYSTEM_PROMPT,
  SUB_LEAF_SYSTEM_PROMPT,
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
  subMaxSteps: 20,
  maxSubCalls: 20,
  maxDepth: 0,
  bashTimeoutMs: 20_000,
  bashOutputByteCap: 8_192,
  subPromptCharCap: 400_000,
} as const;

type ResolvedConfig = Required<
  Omit<RLMEngineConfig, "subModel" | "sandboxRoot" | "onEvent" | "signal">
> &
  Pick<RLMEngineConfig, "subModel" | "sandboxRoot" | "onEvent" | "signal">;

/** State shared across the entire recursion tree (root + all sub-RLMs).
 *  Budgets and usage are global so a depth-2 call can't multiply resources. */
interface SharedState {
  subCalls: number;
  bashCalls: number;
  usage: RLMUsage;
  trace: RLMEvent[];
  emit: (e: RLMEvent) => void;
}

/**
 * An RLM engine. Stateless across invocations — reuse the same engine for
 * many queries. Each `invoke()` creates and disposes a fresh bash sandbox.
 */
export class RLMEngine {
  readonly config: ResolvedConfig;

  constructor(config: RLMEngineConfig) {
    this.config = {
      model: config.model,
      subModel: config.subModel,
      maxSteps: config.maxSteps ?? DEFAULTS.maxSteps,
      subMaxSteps: config.subMaxSteps ?? DEFAULTS.subMaxSteps,
      maxSubCalls: config.maxSubCalls ?? DEFAULTS.maxSubCalls,
      maxDepth: config.maxDepth ?? DEFAULTS.maxDepth,
      bashTimeoutMs: config.bashTimeoutMs ?? DEFAULTS.bashTimeoutMs,
      bashOutputByteCap: config.bashOutputByteCap ?? DEFAULTS.bashOutputByteCap,
      subPromptCharCap: config.subPromptCharCap ?? DEFAULTS.subPromptCharCap,
      sandboxRoot: config.sandboxRoot,
      onEvent: config.onEvent,
      signal: config.signal,
    };
  }

  async invoke(opts: RLMInvokeOptions): Promise<RLMResult> {
    const shared: SharedState = {
      subCalls: 0,
      bashCalls: 0,
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      trace: [],
      emit: () => {},
    };
    shared.emit = (event) => {
      shared.trace.push(event);
      this.config.onEvent?.(event);
    };
    const signal = mergeSignals(opts.signal, this.config.signal);
    const answer = await runRecursive(
      this.config,
      shared,
      { query: opts.query, context: opts.context, additionalInstructions: opts.additionalInstructions ?? "" },
      0,
      signal,
    );
    return {
      answer,
      // `steps` is the step count of the root generateText call only;
      // sub-invocations are counted separately via subCalls.
      steps: shared.trace.filter((e) => e.type === "bash" && e.depth === 0).length +
        (shared.trace.some((e) => e.type === "final" && e.depth === 0) ? 1 : 0),
      subCalls: shared.subCalls,
      bashCalls: shared.bashCalls,
      usage: shared.usage,
      trace: shared.trace,
    };
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

/**
 * Run an RLM invocation at the given depth, reusing the shared budget/usage.
 * Returns the final answer string. Emits `start` + `final` events scoped to
 * the given depth.
 */
async function runRecursive(
  config: ResolvedConfig,
  shared: SharedState,
  opts: { query: string; context: RLMInvokeOptions["context"]; additionalInstructions: string },
  depth: number,
  signal: AbortSignal | undefined,
): Promise<string> {
  if (depth === 0) {
    // Only validate at the top — sub-RLMs are driven by the framework.
    if (!opts.query?.trim()) {
      throw new Error("runRLM: `query` must be a non-empty string.");
    }
  }
  const items = normaliseContext(opts.context);
  if (items.length === 0) {
    throw new Error(
      `runRLM@depth${depth}: context must contain at least one item.`,
    );
  }
  if (items.every((it) => !it.content.length)) {
    throw new Error(`runRLM@depth${depth}: all context items are empty.`);
  }

  const sandbox = await BashSandbox.create({
    root: config.sandboxRoot,
    contextItems: items,
    outputByteCap: config.bashOutputByteCap,
    timeoutMs: config.bashTimeoutMs,
  });

  let finalAnswer: string | null = null;

  shared.emit({ type: "start", query: opts.query, depth });

  try {
    const tools = {
      bash: tool({
        description:
          "Execute a bash command in the persistent REPL. Shell state (vars, functions, cwd) persists across bash calls. Combined stdout+stderr, capped.",
        inputSchema: z.object({
          command: z.string().describe("Bash command to execute."),
        }),
        execute: async ({ command }) => {
          shared.bashCalls++;
          const result = await sandbox.execute(command, signal);
          shared.emit({ type: "bash", command, result, depth });
          return formatBashResult(result);
        },
      }),
      llm: tool({
        description:
          "Recursively call a sub-agent on a focused question and a slice of context you've assembled. " +
          (depth < config.maxDepth
            ? "The sub-agent is ITSELF an RLM — it gets its own bash sandbox with the given context as a file and can grep/slice/recurse further."
            : "At this depth the sub-agent is a plain LLM (no tools) — useful for summarisation/extraction over the snippet."),
        inputSchema: z.object({
          query: z
            .string()
            .describe("The sub-question to answer over the given context slice."),
          context: z
            .string()
            .describe(
              "The context slice (verbatim snippet or assembled buffer) the sub-agent should reason over.",
            ),
        }),
        execute: async ({ query, context }) => {
          if (shared.subCalls >= config.maxSubCalls) {
            return `ERROR: sub-call budget exhausted (${config.maxSubCalls}). Aggregate with what you have and call final.`;
          }
          const totalChars = query.length + context.length;
          if (totalChars > config.subPromptCharCap) {
            return `ERROR: sub-call inputs too large (${totalChars} chars > cap ${config.subPromptCharCap}). Split further.`;
          }
          shared.subCalls++;

          const contextPreview =
            context.length > 200 ? context.slice(0, 200) + "…" : context;

          if (depth < config.maxDepth) {
            // True recursion: sub-agent is a nested RLM with its own sandbox.
            shared.emit({
              type: "sub-start",
              query,
              contextPreview,
              depth: depth + 1,
            });
            const subAnswer = await runRecursive(
              config,
              shared,
              { query, context, additionalInstructions: "" },
              depth + 1,
              signal,
            );
            shared.emit({
              type: "sub-end",
              answer: subAnswer,
              depth: depth + 1,
            });
            return subAnswer;
          } else {
            // Leaf: plain generateText with no tools.
            const { text, usage: subUsage } = await callLeafLM(
              config,
              query,
              context,
              signal,
            );
            if (subUsage) accumulateUsage(shared.usage, subUsage);
            shared.emit({
              type: "sub-llm",
              query,
              contextPreview,
              response: text,
              ...(subUsage ? { usage: subUsage } : {}),
              depth,
            });
            return text;
          }
        },
      }),
      final: tool({
        description:
          "Return the final answer and stop. Call exactly once when you have the answer.",
        inputSchema: z.object({
          answer: z
            .string()
            .describe("The final natural-language answer to the query."),
        }),
        execute: async ({ answer }) => {
          finalAnswer = answer;
          shared.emit({ type: "final", answer, depth });
          return "ok";
        },
      }),
    };

    const stepCap = depth === 0 ? config.maxSteps : config.subMaxSteps;
    // Root runs on the primary model; nested sub-RLMs use subModel when
    // configured (matches the paper's "GPT-5 + GPT-5-mini for recursion").
    const activeModel =
      depth === 0 ? config.model : (config.subModel ?? config.model);
    const result = await generateText({
      model: activeModel,
      system: ROOT_SYSTEM_PROMPT,
      prompt: buildUserPrompt(
        opts.query,
        sandbox.describeContext(),
        opts.additionalInstructions,
      ),
      tools,
      stopWhen: [stepCountIs(stepCap), hasToolCall("final")],
      abortSignal: signal,
    });

    const rootUsage = extractUsage(result.usage);
    if (rootUsage) accumulateUsage(shared.usage, rootUsage);

    if (finalAnswer === null) {
      finalAnswer = result.text?.trim() || "(no answer produced)";
      shared.emit({
        type: "warning",
        message: `RLM@depth${depth} exited without calling final() — used trailing text as the answer.`,
        depth,
      });
      // Emit a final event too so consumers that watch only `final` still see
      // a terminating event on the happy path.
      shared.emit({ type: "final", answer: finalAnswer, depth });
    }
    return finalAnswer;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    shared.emit({ type: "error", error: message, depth });
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

/** Leaf sub-LM call — no tools, just a single generateText. Used at the
 *  deepest recursion level. */
async function callLeafLM(
  config: ResolvedConfig,
  query: string,
  context: string,
  signal?: AbortSignal,
): Promise<{ text: string; usage?: RLMUsage }> {
  const model = config.subModel ?? config.model;
  const res = await generateText({
    model,
    system: SUB_LEAF_SYSTEM_PROMPT,
    prompt: `# Context\n${context}\n\n# Question\n${query}`,
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
  return AbortSignal.any([a, b]);
}

// Re-export token helper for consumers wanting to decide themselves when to
// route a call through the RLM.
export { estimateTokens };
