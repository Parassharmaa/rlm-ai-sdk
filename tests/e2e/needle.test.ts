import "dotenv/config";
import { describe, it, expect } from "vitest";
import { openai } from "@ai-sdk/openai";
import { runRLM } from "../../src/rlm.js";

const HAS_KEY = Boolean(process.env.OPENAI_API_KEY);

const NEEDLE = "The CEO of Acme Corp is named Prof. Zanzibar Montgomery III.";

function buildHaystack(): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < 10; i++) {
    const lines: string[] = [];
    for (let j = 0; j < 150; j++) {
      lines.push(
        `filler line ${i}:${j} — company info about revenue, widgets, and compliance.`,
      );
    }
    if (i === 7) lines[42] = NEEDLE;
    chunks.push(lines.join("\n"));
  }
  return chunks;
}

describe.skipIf(!HAS_KEY)("e2e: needle-in-haystack with real OpenAI", () => {
  it("finds the needle via RLM across ~100k chars of haystack", async () => {
    const result = await runRLM(
      { model: openai("gpt-5-mini"), maxSteps: 30, maxSubCalls: 10 },
      {
        query:
          "Who is the CEO of Acme Corp? Respond with just the person's name.",
        context: buildHaystack(),
      },
    );
    expect(result.answer.toLowerCase()).toContain("zanzibar");
    expect(result.bashCalls).toBeGreaterThan(0);
    expect(result.usage.totalTokens).toBeGreaterThan(0);
  }, 120_000);
});
