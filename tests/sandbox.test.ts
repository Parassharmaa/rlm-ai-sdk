import { describe, expect, it } from "vitest";
import { BashSandbox, normaliseContext } from "../src/sandbox.js";

async function withSandbox<T>(
  opts: Partial<{
    contextItems: Parameters<typeof BashSandbox.create>[0]["contextItems"];
    outputByteCap: number;
    timeoutMs: number;
  }>,
  fn: (sb: BashSandbox) => Promise<T>,
): Promise<T> {
  const sb = await BashSandbox.create({
    contextItems: opts.contextItems ?? [],
    outputByteCap: opts.outputByteCap ?? 8192,
    timeoutMs: opts.timeoutMs ?? 10_000,
  });
  try {
    return await fn(sb);
  } finally {
    await sb.dispose();
  }
}

describe("BashSandbox (persistent REPL)", () => {
  it("executes a simple command and returns stdout", async () => {
    await withSandbox({}, async (sb) => {
      const r = await sb.execute("echo hello");
      expect(r.exitCode).toBe(0);
      expect(r.stdout).toBe("hello");
      expect(r.timedOut).toBe(false);
      expect(r.truncated).toBe(false);
    });
  });

  it("persists shell variables across calls (the REPL point)", async () => {
    await withSandbox({}, async (sb) => {
      const r1 = await sb.execute("x=42");
      expect(r1.exitCode).toBe(0);
      const r2 = await sb.execute('echo "x is $x"');
      expect(r2.stdout).toBe("x is 42");
    });
  });

  it("persists shell functions across calls", async () => {
    await withSandbox({}, async (sb) => {
      await sb.execute("greet() { echo \"hello $1\"; }");
      const r = await sb.execute("greet world");
      expect(r.stdout).toBe("hello world");
    });
  });

  it("persists cwd changes across calls", async () => {
    await withSandbox({}, async (sb) => {
      await sb.execute('mkdir -p sub && cd sub');
      const r = await sb.execute("basename $(pwd)");
      expect(r.stdout).toBe("sub");
    });
  });

  it("exposes context items as files and env vars", async () => {
    await withSandbox(
      {
        contextItems: [
          { id: "foo", content: "hello foo" },
          { id: "bar", content: "hello bar" },
        ],
      },
      async (sb) => {
        const r1 = await sb.execute('cat "$RLM_CTX_FOO"');
        expect(r1.exitCode).toBe(0);
        expect(r1.stdout).toBe("hello foo");

        const r2 = await sb.execute('ls "$RLM_CTX_DIR"');
        expect(r2.stdout.split("\n").filter(Boolean).sort()).toEqual([
          "bar.txt",
          "foo.txt",
        ]);
      },
    );
  });

  it("merges stderr into stdout", async () => {
    await withSandbox({}, async (sb) => {
      const r = await sb.execute('echo stdout1 && echo stderr1 1>&2 && echo stdout2');
      expect(r.exitCode).toBe(0);
      // All three should appear (order is whatever bash flushes; usually as written).
      expect(r.stdout).toContain("stdout1");
      expect(r.stdout).toContain("stderr1");
      expect(r.stdout).toContain("stdout2");
      expect(r.stderr).toBe("");
    });
  });

  it("captures non-zero exit codes from subshells", async () => {
    await withSandbox({}, async (sb) => {
      const r = await sb.execute("(exit 7)");
      expect(r.exitCode).toBe(7);
      // REPL should still be alive — the subshell exited, not our bash.
      const r2 = await sb.execute("echo alive");
      expect(r2.stdout).toBe("alive");
    });
  });

  it("marks the sandbox dead if the user issues `exit`", async () => {
    await withSandbox({}, async (sb) => {
      await sb.execute("exit 9");
      const r = await sb.execute("echo never");
      expect(r.exitCode).toBeGreaterThanOrEqual(128);
      expect(r.stderr).toContain("sandbox died");
    });
  });

  it("marks sandbox dead on timeout and dead-results subsequent calls", async () => {
    await withSandbox({ timeoutMs: 200 }, async (sb) => {
      const r = await sb.execute("sleep 5 && echo done");
      expect(r.timedOut).toBe(true);
      expect(r.exitCode).toBe(124);
      const r2 = await sb.execute("echo alive");
      expect(r2.stderr).toContain("sandbox died");
    });
  });

  it("continues after a command that returns non-zero (but doesn't exit)", async () => {
    await withSandbox({}, async (sb) => {
      const r1 = await sb.execute("false");
      expect(r1.exitCode).toBe(1);
      const r2 = await sb.execute("echo alive");
      expect(r2.exitCode).toBe(0);
      expect(r2.stdout).toBe("alive");
    });
  });

  it("truncates output that exceeds the byte cap", async () => {
    await withSandbox({ outputByteCap: 256 }, async (sb) => {
      const r = await sb.execute("yes x | head -c 2000");
      expect(r.truncated).toBe(true);
      expect(r.stdout.length).toBeLessThanOrEqual(256);
    });
  });


  it("handles heredocs without hanging the done marker", async () => {
    await withSandbox({}, async (sb) => {
      const r = await sb.execute(
        "cat <<'EOF'\nline1\nline2\nEOF",
      );
      expect(r.exitCode).toBe(0);
      expect(r.stdout).toBe("line1\nline2");
    });
  });

  it("rejects unsafe context ids", async () => {
    await expect(
      BashSandbox.create({
        contextItems: [{ id: "../etc/passwd", content: "bad" }],
        outputByteCap: 1024,
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/Invalid context id/);
  });

  it("cleans up the temp workdir when setup fails", async () => {
    // List rlm-* dirs before and after a failing create to verify cleanup.
    const { readdir } = await import("node:fs/promises");
    const os = await import("node:os");
    const tmpRoot = os.tmpdir();
    const before = (await readdir(tmpRoot)).filter((n) => n.startsWith("rlm-"));
    await expect(
      BashSandbox.create({
        contextItems: [{ id: "../bad", content: "x" }],
        outputByteCap: 1024,
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/Invalid context id/);
    const after = (await readdir(tmpRoot)).filter((n) => n.startsWith("rlm-"));
    expect(after.length).toBe(before.length);
  });

  it("redacts credential-ish env vars from bash env", async () => {
    const cases: Array<[string, string]> = [
      ["OPENAI_API_KEY", "sk-openai"],
      ["ANTHROPIC_API_KEY", "sk-ant"],
      ["HUGGINGFACE_API_KEY", "hf_x"],
      ["MY_DB_PASSWORD", "p455"],
      ["AWS_ACCESS_KEY_ID", "AKIAX"],
      ["GITHUB_TOKEN", "ghp_x"],
    ];
    const prev: Record<string, string | undefined> = {};
    for (const [k, v] of cases) {
      prev[k] = process.env[k];
      process.env[k] = v;
    }
    try {
      await withSandbox({}, async (sb) => {
        for (const [k] of cases) {
          const r = await sb.execute(`echo "[$${k}]"`);
          expect(r.stdout, `expected ${k} redacted`).toBe("[]");
        }
      });
    } finally {
      for (const [k] of cases) {
        if (prev[k] === undefined) delete process.env[k];
        else process.env[k] = prev[k];
      }
    }
  });

  it("does not redact non-credential env vars", async () => {
    const prev = process.env.RLM_TEST_NON_CRED;
    process.env.RLM_TEST_NON_CRED = "visible";
    try {
      await withSandbox({}, async (sb) => {
        const r = await sb.execute('echo "[$RLM_TEST_NON_CRED]"');
        expect(r.stdout).toBe("[visible]");
      });
    } finally {
      if (prev === undefined) delete process.env.RLM_TEST_NON_CRED;
      else process.env.RLM_TEST_NON_CRED = prev;
    }
  });

  it("serializes concurrent execute calls", async () => {
    await withSandbox({}, async (sb) => {
      // Two parallel commands each need ~200ms of wall time. If the sandbox
      // serialises them, total elapsed should be ≥400ms. If it didn't, we'd
      // see clobbered output because the REPL is single-threaded.
      const [a, b] = await Promise.all([
        sb.execute('sleep 0.15 && echo a'),
        sb.execute('sleep 0.15 && echo b'),
      ]);
      expect(a.stdout).toBe("a");
      expect(b.stdout).toBe("b");
    });
  });

  it("is aborted by AbortSignal", async () => {
    await withSandbox({}, async (sb) => {
      const ac = new AbortController();
      setTimeout(() => ac.abort(), 100);
      const r = await sb.execute("sleep 5 && echo done", ac.signal);
      expect(r.stdout).not.toContain("done");
    });
  });
});

describe("normaliseContext", () => {
  it("wraps a single string as one item", () => {
    expect(normaliseContext("hello")).toEqual([
      { id: "context", content: "hello" },
    ]);
  });
  it("returns [] for empty string", () => {
    expect(normaliseContext("")).toEqual([]);
  });
  it("names string arrays chunk_<i>", () => {
    expect(normaliseContext(["a", "b"])).toEqual([
      { id: "chunk_0", content: "a" },
      { id: "chunk_1", content: "b" },
    ]);
  });
  it("passes ContextItem arrays through", () => {
    const items = [{ id: "x", content: "y" }];
    expect(normaliseContext(items)).toBe(items);
  });
});
