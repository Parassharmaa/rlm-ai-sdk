import "dotenv/config";
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

for (const id of ["gpt-5", "gpt-5-mini"]) {
  try {
    const r = await generateText({
      model: openai(id as never),
      prompt: "Say 'ok' and nothing else.",
    });
    console.log(
      `${id}: OK → "${r.text.trim()}" usage=${JSON.stringify(r.usage)}`,
    );
  } catch (e) {
    console.log(`${id}: FAIL → ${(e as Error).message.slice(0, 200)}`);
  }
}
