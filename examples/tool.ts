/**
 * Tool example — give a regular AI SDK agent an `rlm` tool so it can
 * recurse on large blobs on demand.
 *
 * Run:
 *   pnpm example:tool
 */
import "dotenv/config";
import { generateText, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";
import { rlmTool } from "../src/index.js";

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    console.error("Set OPENAI_API_KEY (or symlink .env) to run this example.");
    process.exit(1);
  }

  const model = openai("gpt-5-mini");

  const bigDocument = Array.from({ length: 800 }, (_, i) => {
    if (i === 537)
      return `Line ${i}: The secret launch code is ALPHA-TANGO-7729.`;
    return `Line ${i}: routine log entry, nothing of note.`;
  }).join("\n");

  const result = await generateText({
    model,
    system:
      "You are a research assistant. When asked to find information in a large document, use the `rlm` tool with the full document as the context.",
    prompt: [
      "Here is a document. Find the launch code and report it.",
      "Use the `rlm` tool — do NOT try to search by hand.",
      "Document follows between <DOC> tags.",
      "<DOC>",
      bigDocument,
      "</DOC>",
    ].join("\n"),
    tools: {
      rlm: rlmTool({ model, maxSteps: 20 }),
    },
    stopWhen: stepCountIs(6),
  });

  console.log("=== Agent answer ===");
  console.log(result.text);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
