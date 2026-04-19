import type { LanguageModel } from "ai";

/**
 * A piece of input to be made available to the RLM as part of the context.
 * The root LM can reference these by `id` (e.g. "chunk_0", "doc_readme") in
 * bash — each item is written to a separate file in the sandbox workdir
 * under the name `<id>.txt`, and the path is exposed via env var
 * `RLM_CTX_<ID_UPPER>`.
 */
export interface ContextItem {
  id: string;
  content: string;
  /** Optional human-readable description shown to the root LM. */
  description?: string;
}

/** Input context: either a single string, an array of strings, or named items. */
export type RLMContext = string | string[] | ContextItem[];

/**
 * Options for a single RLM invocation.
 */
export interface RLMInvokeOptions {
  /** Natural-language query the RLM is trying to answer. */
  query: string;
  /** The oversized context the RLM explores. */
  context: RLMContext;
  /**
   * Extra system instructions appended after the RLM root prompt. Use this
   * to give task-specific hints or constraints.
   */
  additionalInstructions?: string;
  /** Override for this specific call. */
  signal?: AbortSignal;
}

/** Bash execution result. */
export interface BashResult {
  stdout: string;
  stderr: string;
  exitCode: number;
  /** True if output was truncated to fit the configured byte cap. */
  truncated: boolean;
  /** Execution wall time in ms. */
  durationMs: number;
  /** True if the command hit the per-call timeout. */
  timedOut: boolean;
}

/** Aggregated token usage across an RLM run. All fields optional because
 *  some models/providers don't report usage. */
export interface RLMUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

/** A single event emitted during an RLM run. Useful for tracing / UIs.
 *  `depth` is 0 for the root, 1 for a sub-RLM, 2 for a sub-sub-RLM, etc. */
export type RLMEvent =
  | { type: "start"; query: string; depth: number }
  | { type: "bash"; command: string; result: BashResult; depth: number }
  | {
      /** Leaf sub-LLM call (recursion bottomed out at maxDepth). */
      type: "sub-llm";
      query: string;
      contextPreview: string;
      response: string;
      usage?: RLMUsage;
      depth: number;
    }
  | {
      /** Nested sub-RLM call (depth < maxDepth) — its own inner trace follows
       *  until the matching "sub-end" event. */
      type: "sub-start";
      query: string;
      contextPreview: string;
      depth: number;
    }
  | {
      type: "sub-end";
      answer: string;
      depth: number;
    }
  | { type: "final"; answer: string; depth: number }
  | { type: "error"; error: string; depth: number };

/** Result of a single RLM invocation. */
export interface RLMResult {
  answer: string;
  steps: number;
  subCalls: number;
  bashCalls: number;
  /** Summed token usage across root + all sub-calls (where reported). */
  usage: RLMUsage;
  /** Full chronological trace — always populated for debuggability. */
  trace: RLMEvent[];
}

/**
 * Configuration for an RLM engine. The engine is stateless between `invoke`
 * calls — each call spins up its own sandbox. Model + limits are shared.
 */
export interface RLMEngineConfig {
  /**
   * The AI SDK language model to use for the root LM. The same model is
   * reused for recursive sub-calls unless `subModel` is provided.
   */
  model: LanguageModel;
  /** Optional separate model for sub-calls (e.g. a cheaper/faster model). */
  subModel?: LanguageModel;
  /** Max steps the root LM may take (bash + llm calls). Default: 40. */
  maxSteps?: number;
  /** Max steps the LM inside a nested sub-RLM may take. Default: 20. */
  subMaxSteps?: number;
  /**
   * Max total recursive sub-LM calls, summed across the entire tree of
   * invocations (root + every nested sub-RLM). Default: 20.
   */
  maxSubCalls?: number;
  /**
   * Max recursion depth for the `llm` tool.
   * - `maxDepth=0` (default): every `llm(query, context)` call is a leaf —
   *   a plain generateText with no inner bash sandbox. This matches the
   *   paper's `llm_query` semantics for typical summarisation/extraction.
   * - `maxDepth=1`: sub-calls are themselves RLMs with their own bash
   *   sandbox, and can partition/grep within their slice.
   * - `maxDepth=2+`: multi-layer recursion (expensive — each level spawns
   *   its own sandbox and burns steps).
   */
  maxDepth?: number;
  /** Per-bash-command timeout in ms. Default: 20_000. */
  bashTimeoutMs?: number;
  /** Cap on bash stdout+stderr bytes returned per call. Default: 8_192. */
  bashOutputByteCap?: number;
  /** Cap on chars sent into a single recursive sub-LM call. Default: 400_000. */
  subPromptCharCap?: number;
  /** Sandbox temp-root override. Defaults to os.tmpdir(). */
  sandboxRoot?: string;
  /** Optional event callback for streaming / UI. */
  onEvent?: (event: RLMEvent) => void;
  /** Default abort signal applied to all invocations. */
  signal?: AbortSignal;
}
