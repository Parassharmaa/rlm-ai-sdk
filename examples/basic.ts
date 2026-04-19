/**
 * Basic RLM example — ask a question over a large synthetic context.
 *
 * Run:
 *   pnpm example:basic
 */
import "dotenv/config";
import { openai } from "@ai-sdk/openai";
import { runRLM } from "../src/index.js";

const NEEDLE = "The CEO of Acme Corp is named Prof. Zanzibar Montgomery III.";

function buildHaystack(): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < 20; i++) {
    const lines: string[] = [];
    for (let j = 0; j < 200; j++) {
      lines.push(
        `filler line ${i}:${j} — some company info about quarterly revenue, widgets, and compliance.`,
      );
    }
    if (i === 13) lines[73] = NEEDLE;
    chunks.push(lines.join("\n"));
  }
  return chunks;
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    console.error("Set OPENAI_API_KEY (or symlink .env) to run this example.");
    process.exit(1);
  }

  const context = buildHaystack();
  const totalChars = context.reduce((n, c) => n + c.length, 0);
  console.log(
    `Context: ${context.length} chunks, ${totalChars.toLocaleString()} chars.`,
  );

  const model = openai("gpt-5-mini");
  const result = await runRLM(
    {
      model,
      maxSteps: 30,
      maxSubCalls: 10,
      onEvent: (e) => {
        if (e.type === "bash") {
          console.log(`[bash] ${e.result.durationMs}ms exit=${e.result.exitCode}`);
        } else if (e.type === "sub-llm") {
          console.log(`[sub-llm] ${e.response.slice(0, 80)}`);
        } else if (e.type === "final") {
          console.log(`[final]`);
        } else if (e.type === "error") {
          console.log(`[error] ${e.error}`);
        }
      },
    },
    {
      query: "Who is the CEO of Acme Corp? Respond with the person's name only.",
      context,
    },
  );

  console.log("\n=== Answer ===");
  console.log(result.answer);
  console.log(
    `\nSteps=${result.steps} bashCalls=${result.bashCalls} subCalls=${result.subCalls}`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
