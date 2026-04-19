import type { LanguageModelMiddleware } from "ai";
import { runRLM } from "./rlm.js";
import { estimateTokens } from "./sandbox.js";
import type { RLMEngineConfig, RLMResult } from "./types.js";

/**
 * Options for the RLM middleware.
 */
export interface RLMMiddlewareOptions
  extends Omit<RLMEngineConfig, "model"> {
  /**
   * Model used for the RLM root LM and (unless overridden) sub-LMs. If
   * omitted, the middleware will route through the underlying model being
   * wrapped — i.e. the same model handles both the root loop and sub-calls.
   */
  model?: RLMEngineConfig["model"];
  /**
   * Activation threshold. When the estimated input token count exceeds this,
   * the call is routed through the RLM instead of going straight to the
   * underlying model. Default: 32_000 tokens.
   */
  thresholdTokens?: number;
  /**
   * Predicate override: return true to route through RLM, false to pass
   * through. Takes precedence over `thresholdTokens` when provided.
   */
  shouldRoute?: (params: WrapGenerateParams) => boolean;
  /**
   * Called when an RLM run completes. Useful for logging / metrics.
   */
  onResult?: (result: RLMResult) => void;
}

const DEFAULT_THRESHOLD_TOKENS = 32_000;

/**
 * AI SDK middleware that transparently routes oversized prompts through an
 * RLM. Small prompts pass through unchanged.
 *
 * @example
 *   import { wrapLanguageModel } from "ai";
 *   import { openai } from "@ai-sdk/openai";
 *   import { createRLMMiddleware } from "rlm-ai-sdk";
 *
 *   const model = wrapLanguageModel({
 *     model: openai("gpt-4o-mini"),
 *     middleware: createRLMMiddleware({ thresholdTokens: 20_000 }),
 *   });
 */
type WrapGenerateArg = Parameters<
  NonNullable<LanguageModelMiddleware["wrapGenerate"]>
>[0];
type WrapGenerateParams = WrapGenerateArg["params"];
type WrapGenerateResult = Awaited<
  ReturnType<NonNullable<LanguageModelMiddleware["wrapGenerate"]>>
>;
type WrapStreamResult = Awaited<
  ReturnType<NonNullable<LanguageModelMiddleware["wrapStream"]>>
>;

export function createRLMMiddleware(
  options: RLMMiddlewareOptions = {},
): LanguageModelMiddleware {
  const threshold = options.thresholdTokens ?? DEFAULT_THRESHOLD_TOKENS;

  const decideRoute = (params: WrapGenerateParams) =>
    options.shouldRoute
      ? options.shouldRoute(params)
      : estimatePromptTokens(params) > threshold;

  return {
    specificationVersion: "v3",
    async wrapGenerate({ doGenerate, params, model: underlyingModel }) {
      if (!decideRoute(params)) return await doGenerate();
      const extracted = extractQueryAndContext(params);
      if (!extracted) return await doGenerate();
      const rlmResult = await runRLM(
        { ...options, model: options.model ?? underlyingModel },
        {
          query: extracted.query,
          context: extracted.context,
          signal: params.abortSignal,
        },
      );
      options.onResult?.(rlmResult);
      return synthesiseGenerateResult(rlmResult);
    },
    async wrapStream({ doStream, params, model: underlyingModel }) {
      if (!decideRoute(params)) return await doStream();
      const extracted = extractQueryAndContext(params);
      if (!extracted) return await doStream();
      const rlmResult = await runRLM(
        { ...options, model: options.model ?? underlyingModel },
        {
          query: extracted.query,
          context: extracted.context,
          signal: params.abortSignal,
        },
      );
      options.onResult?.(rlmResult);
      return synthesiseStreamResult(rlmResult);
    },
  };
}

/** Rough token estimate across all prompt parts. */
function estimatePromptTokens(params: WrapGenerateParams): number {
  let total = 0;
  for (const msg of params.prompt) {
    if (typeof msg.content === "string") {
      total += estimateTokens(msg.content);
    } else if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if ("text" in part && typeof part.text === "string") {
          total += estimateTokens(part.text);
        }
      }
    }
  }
  return total;
}

/**
 * Pull a sensible (query, context) pair out of an AI SDK call. The last
 * user message becomes the query; everything prior (system + earlier
 * messages) becomes the context. This is a heuristic — callers who want
 * more control should use `runRLM` directly or supply `shouldRoute`.
 */
function extractQueryAndContext(
  params: WrapGenerateParams,
): { query: string; context: string[] } | null {
  const messages = params.prompt;
  if (messages.length === 0) return null;

  let query = "";
  const contextChunks: string[] = [];

  // Walk backwards to find the last user message — that's the query.
  let queryIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      queryIdx = i;
      break;
    }
  }
  if (queryIdx === -1) return null;

  const queryMsg = messages[queryIdx];
  query = stringifyContent(queryMsg.content);
  if (!query.trim()) return null;

  for (let i = 0; i < messages.length; i++) {
    if (i === queryIdx) continue;
    const text = stringifyContent(messages[i].content);
    if (text.trim()) {
      contextChunks.push(`[${messages[i].role}]\n${text}`);
    }
  }

  return { query, context: contextChunks.length > 0 ? contextChunks : [query] };
}

function stringifyContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (part && typeof part === "object" && "text" in part) {
        return String((part as { text: unknown }).text ?? "");
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

const makeUsage = (result: RLMResult) => ({
  inputTokens: {
    total: result.usage.inputTokens || undefined,
    noCache: undefined,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: {
    total: result.usage.outputTokens || undefined,
    text: result.usage.outputTokens || undefined,
    reasoning: undefined,
  },
});

const makeMetadata = (result: RLMResult) => ({
  "rlm-ai-sdk": {
    steps: result.steps,
    subCalls: result.subCalls,
    bashCalls: result.bashCalls,
    inputTokens: result.usage.inputTokens,
    outputTokens: result.usage.outputTokens,
    totalTokens: result.usage.totalTokens,
  },
});

/** Package the RLM answer into the shape AI SDK's wrapGenerate expects. */
function synthesiseGenerateResult(result: RLMResult): WrapGenerateResult {
  const out: WrapGenerateResult = {
    content: [{ type: "text", text: result.answer }],
    finishReason: { unified: "stop", raw: undefined },
    usage: makeUsage(result),
    warnings: [],
    providerMetadata: makeMetadata(result),
  };
  return out;
}

/** Package the RLM answer into the shape AI SDK's wrapStream expects —
 *  a ReadableStream of LanguageModelV3StreamPart with a single text block
 *  followed by a finish event. */
function synthesiseStreamResult(result: RLMResult): WrapStreamResult {
  const usage = makeUsage(result);
  const providerMetadata = makeMetadata(result);
  const id = "rlm-answer";
  type Part =
    | { type: "text-start"; id: string }
    | { type: "text-delta"; id: string; delta: string }
    | { type: "text-end"; id: string }
    | {
        type: "finish";
        finishReason: { unified: "stop"; raw: undefined };
        usage: ReturnType<typeof makeUsage>;
        providerMetadata: ReturnType<typeof makeMetadata>;
      };
  const parts: Part[] = [
    { type: "text-start", id },
    { type: "text-delta", id, delta: result.answer },
    { type: "text-end", id },
    {
      type: "finish",
      finishReason: { unified: "stop", raw: undefined },
      usage,
      providerMetadata,
    },
  ];
  const stream = new ReadableStream({
    start(controller) {
      for (const p of parts) controller.enqueue(p);
      controller.close();
    },
  });
  return { stream } as WrapStreamResult;
}
