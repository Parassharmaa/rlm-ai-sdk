import { randomBytes } from "node:crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { BashResult, ContextItem } from "./types.js";

/** Env vars whose names match this regex are blanked before bash sees them. */
const CREDENTIAL_ENV_RE =
  /(?:_API_KEY|_SECRET|_TOKEN|_PASSWORD|_PASS|_CREDENTIALS|SESSION_TOKEN|ACCESS_KEY)$|^AWS_|^AZURE_|^GCP_|^GOOGLE_APPLICATION_CREDENTIALS$/i;

/**
 * Persistent bash REPL sandbox. One `bash` process lives for the lifetime
 * of the sandbox; the LM sends commands over stdin and receives output on
 * stdout. Shell state — variables, functions, cwd, aliases — **persists**
 * across `execute` calls, matching the paper's Jupyter paradigm with bash
 * as the runtime.
 *
 * Protocol:
 *   - stderr is merged into stdout via `exec 2>&1` at startup.
 *   - After each user command, a shell function `__rlm_done <exit>` emits a
 *     unique sentinel line containing the exit code. We read stdout until we
 *     see that sentinel.
 *   - The sentinel nonce is random per sandbox so user output can't forge it.
 *   - Per-command timeout: SIGKILL the bash process. The command (and any
 *     surviving shell state) is lost; the sandbox is marked dead; subsequent
 *     calls return a dead-result instead of hanging. The RLM loop treats this
 *     as a recoverable signal — the model can still call `final`.
 *   - Calls are serialized — a persistent REPL is single-threaded.
 */
export class BashSandbox {
  readonly workdir: string;
  readonly contextItems: ContextItem[];
  private readonly outputByteCap: number;
  private readonly timeoutMs: number;
  private readonly nonce: string;
  private readonly doneMarker: string;
  private proc: ChildProcessWithoutNullStreams | null = null;
  private disposed = false;
  private dead = false;
  private deadReason: string | null = null;
  private queue: Promise<unknown> = Promise.resolve();

  private constructor(
    workdir: string,
    contextItems: ContextItem[],
    outputByteCap: number,
    timeoutMs: number,
  ) {
    this.workdir = workdir;
    this.contextItems = contextItems;
    this.outputByteCap = outputByteCap;
    this.timeoutMs = timeoutMs;
    this.nonce = randomBytes(8).toString("hex");
    this.doneMarker = `__RLM_DONE_${this.nonce}__:`;
  }

  static async create(opts: {
    root?: string;
    contextItems: ContextItem[];
    outputByteCap: number;
    timeoutMs: number;
  }): Promise<BashSandbox> {
    const root = opts.root ?? tmpdir();
    const workdir = await mkdtemp(join(root, "rlm-"));
    const ctxDir = join(workdir, "ctx");
    await mkdir(ctxDir, { recursive: true });

    for (const item of opts.contextItems) {
      if (!/^[a-zA-Z0-9_.-]+$/.test(item.id)) {
        throw new Error(
          `Invalid context id "${item.id}" — must match [a-zA-Z0-9_.-]+`,
        );
      }
      await writeFile(join(ctxDir, `${item.id}.txt`), item.content, "utf8");
    }
    const sandbox = new BashSandbox(
      workdir,
      opts.contextItems,
      opts.outputByteCap,
      opts.timeoutMs,
    );
    await sandbox.boot();
    return sandbox;
  }

  /** Human-readable catalog of context items for the root prompt. */
  describeContext(): string {
    if (this.contextItems.length === 0) return "(empty context)";
    const lines: string[] = [];
    for (const item of this.contextItems) {
      const bytes = Buffer.byteLength(item.content, "utf8");
      const envName = `RLM_CTX_${item.id.toUpperCase().replace(/[^A-Z0-9]/g, "_")}`;
      lines.push(
        `- id=${item.id} (${bytes} bytes) path=$${envName} ${
          item.description ? `— ${item.description}` : ""
        }`,
      );
    }
    return lines.join("\n");
  }

  private async boot(): Promise<void> {
    const env: Record<string, string> = {
      ...process.env,
      RLM_WORKDIR: this.workdir,
      RLM_CTX_DIR: join(this.workdir, "ctx"),
      // Keep bash quiet / predictable.
      PS1: "",
      PS2: "",
      PS3: "",
      PS4: "",
      TERM: "dumb",
      HISTFILE: "/dev/null",
    };
    for (const item of this.contextItems) {
      const envName = `RLM_CTX_${item.id.toUpperCase().replace(/[^A-Z0-9]/g, "_")}`;
      env[envName] = join(this.workdir, "ctx", `${item.id}.txt`);
    }
    // Blank credential-ish env vars.
    for (const key of Object.keys(env)) {
      if (CREDENTIAL_ENV_RE.test(key)) env[key] = "";
    }

    const proc = spawn(
      "bash",
      // --noprofile --norc: don't load user rc files. Non-interactive: bash
      // reads commands from stdin line-by-line. Non-zero exits from commands
      // do NOT terminate bash — only explicit `exit`, signals, or EOF do.
      ["--noprofile", "--norc"],
      {
        cwd: this.workdir,
        env,
        stdio: ["pipe", "pipe", "pipe"],
      },
    );
    this.proc = proc;

    proc.on("exit", (code, signal) => {
      this.dead = true;
      this.deadReason = `bash exited code=${code} signal=${signal ?? "none"}`;
    });
    proc.on("error", (err) => {
      this.dead = true;
      this.deadReason = `bash error: ${err.message}`;
    });

    // Bootstrap: merge stderr into stdout, define done helper.
    // The \n before the marker guarantees it lands at line start even if the
    // preceding command's output didn't end with a newline.
    const bootstrap =
      [
        "set +o history",
        "exec 2>&1",
        `__rlm_done() { printf '\\n${this.doneMarker}%d\\n' "$1"; }`,
        // Sync-ping so we know bash is alive before the first real command.
        "__rlm_done 0",
      ].join("\n") + "\n";

    await this.runRaw(bootstrap, /*isBootstrap*/ true);
  }

  /**
   * Serialized execute — the persistent REPL is single-threaded.
   */
  async execute(command: string, signal?: AbortSignal): Promise<BashResult> {
    if (this.disposed) throw new Error("sandbox already disposed");
    const release = await this.acquire();
    try {
      if (this.dead) {
        return this.makeDeadResult(command);
      }
      return await this.runCommand(command, signal);
    } finally {
      release();
    }
  }

  private acquire(): Promise<() => void> {
    let release!: () => void;
    const next = new Promise<void>((r) => {
      release = r;
    });
    const prev = this.queue;
    this.queue = prev.then(() => next);
    return prev.then(() => release);
  }

  private makeDeadResult(_command: string): BashResult {
    return {
      stdout: "",
      stderr: `sandbox died: ${this.deadReason ?? "unknown"}`,
      exitCode: 137,
      truncated: false,
      durationMs: 0,
      timedOut: false,
    };
  }

  /** Send a block of bash (already terminated) and read until done marker. */
  private runRaw(
    block: string,
    isBootstrap: boolean,
  ): Promise<BashResult> {
    return this.runImpl(block, isBootstrap);
  }

  private async runCommand(
    command: string,
    signal?: AbortSignal,
  ): Promise<BashResult> {
    // Wrap the user command so that __rlm_done always fires — even if the
    // command ends with an unbalanced continuation, using `{ ...; }` means
    // bash will reject the whole block on syntax error rather than stall
    // waiting for more input. We append on a new line so heredocs inside the
    // user command close before __rlm_done is seen.
    const block = `${command}\n__rlm_done $?\n`;
    return await this.runImpl(block, false, signal);
  }

  private runImpl(
    block: string,
    isBootstrap: boolean,
    signal?: AbortSignal,
  ): Promise<BashResult> {
    const proc = this.proc;
    if (!proc || this.dead) {
      return Promise.resolve(this.makeDeadResult(block));
    }

    return new Promise<BashResult>((resolve) => {
      const start = Date.now();
      // We keep a small sliding tail buffer for scanning the done marker
      // even after we've hit the output cap, so long-running commands with
      // huge output still return cleanly.
      const TAIL_KEEP = this.doneMarker.length + 32;
      let display = "";
      let displayBytes = 0;
      let tail = "";
      let truncated = false;
      let settled = false;
      let timedOut = false;

      const cleanup = () => {
        proc.stdout.off("data", onData);
        proc.off("exit", onExit);
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
      };

      const onExit = (code: number | null, sigName: NodeJS.Signals | null) => {
        if (settled) return;
        this.dead = true;
        this.deadReason = `bash exited code=${code} signal=${sigName ?? "none"}`;
        finishWith(
          removeMarkerTail(display, this.doneMarker),
          code ?? (sigName ? 128 : 1),
        );
      };

      const finishWith = (stdout: string, exitCode: number) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve({
          stdout,
          stderr: "", // merged into stdout via `exec 2>&1`
          exitCode,
          truncated,
          durationMs: Date.now() - start,
          timedOut,
        });
      };

      const onData = (chunk: Buffer) => {
        const text = chunk.toString("utf8");
        // Always scan the tail for the done marker so we notice it even
        // after the displayed output has been capped.
        tail = (tail + text).slice(-Math.max(TAIL_KEEP, text.length + TAIL_KEEP));
        // Append to displayed output up to the cap.
        const room = this.outputByteCap - displayBytes;
        if (text.length > room) {
          if (room > 0) {
            display += text.slice(0, room);
            displayBytes = this.outputByteCap;
          }
          truncated = true;
        } else {
          display += text;
          displayBytes += text.length;
        }
        const markerIdx = tail.indexOf(this.doneMarker);
        if (markerIdx >= 0) {
          const tailAfter = tail.slice(markerIdx + this.doneMarker.length);
          const m = tailAfter.match(/^(-?\d+)/);
          const exitCode = m ? parseInt(m[1] as string, 10) : 0;
          // Remove any trace of the marker from displayed output if it made
          // it in there before the cap triggered.
          const cleanDisplay = removeMarkerTail(display, this.doneMarker);
          finishWith(isBootstrap ? "" : cleanDisplay, exitCode);
        }
      };

      const onTimeout = () => {
        timedOut = true;
        this.dead = true;
        this.deadReason = "command timed out";
        try {
          proc.kill("SIGKILL");
        } catch {
          /* ignore */
        }
        finishWith(removeMarkerTail(display, this.doneMarker), 124);
      };

      const timer = setTimeout(onTimeout, this.timeoutMs);

      const onAbort = () => {
        if (settled) return;
        this.dead = true;
        this.deadReason = "aborted by caller";
        try {
          proc.kill("SIGKILL");
        } catch {
          /* ignore */
        }
        finishWith(removeMarkerTail(display, this.doneMarker), 137);
      };

      if (signal) {
        if (signal.aborted) {
          onAbort();
          return;
        }
        signal.addEventListener("abort", onAbort, { once: true });
      }

      proc.stdout.on("data", onData);
      proc.on("exit", onExit);
      proc.stdin.write(block, (err) => {
        if (err && !settled) {
          this.dead = true;
          this.deadReason = `stdin write failed: ${err.message}`;
          finishWith("", 127);
        }
      });
    });
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    if (this.proc && !this.proc.killed) {
      try {
        this.proc.stdin.end();
      } catch {
        /* ignore */
      }
      try {
        this.proc.kill("SIGKILL");
      } catch {
        /* ignore */
      }
    }
    await rm(this.workdir, { recursive: true, force: true });
  }
}

function stripTrailingNewline(s: string): string {
  return s.replace(/\n+$/, "");
}

/** Strip the done marker (and anything after) + its leading newline from the
 *  displayed output. The marker is emitted by `printf '\n<marker>%d\n'`, so
 *  it sits at the tail as `\n<marker><digits>\n`. */
function removeMarkerTail(output: string, marker: string): string {
  const idx = output.indexOf(marker);
  if (idx === -1) return stripTrailingNewline(output);
  // Drop the leading \n the marker was printed with, if present.
  const trimmed =
    idx > 0 && output[idx - 1] === "\n"
      ? output.slice(0, idx - 1)
      : output.slice(0, idx);
  return stripTrailingNewline(trimmed);
}

/** Normalise a user-provided context into a stable list of ContextItems. */
export function normaliseContext(
  ctx: string | string[] | ContextItem[],
): ContextItem[] {
  if (typeof ctx === "string") {
    return ctx ? [{ id: "context", content: ctx }] : [];
  }
  if (Array.isArray(ctx)) {
    if (ctx.length === 0) return [];
    if (typeof ctx[0] === "string") {
      return (ctx as string[]).map((content, i) => ({
        id: `chunk_${i}`,
        content,
      }));
    }
    return ctx as ContextItem[];
  }
  return [];
}

/** Build a token-ish estimate (chars / 4) — good enough for budgeting. */
export function estimateTokens(s: string): number {
  return Math.ceil(s.length / 4);
}
