/**
 * LongBench-v2 Code Repository Understanding subset loader.
 * Dataset: https://huggingface.co/datasets/zai-org/LongBench-v2
 * Paper (2025): https://arxiv.org/abs/2412.15204
 *
 * Fields: _id, domain, sub_domain, difficulty, length, question,
 *         choice_A..D, answer (letter), context.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

export interface LongBenchItem {
  _id: string;
  domain: string;
  sub_domain: string;
  difficulty: "easy" | "hard";
  length: "short" | "medium" | "long";
  question: string;
  choice_A: string;
  choice_B: string;
  choice_C: string;
  choice_D: string;
  answer: "A" | "B" | "C" | "D";
  context: string;
}

export function loadCodeQA(options: {
  path?: string;
  maxContextTokens?: number;
  limit?: number;
}): LongBenchItem[] {
  const path = options.path ?? join("bench", "data", "longbench-v2.json");
  const all = JSON.parse(readFileSync(path, "utf8")) as LongBenchItem[];
  let items = all.filter((r) => r.domain === "Code Repository Understanding");
  if (options.maxContextTokens !== undefined) {
    const maxChars = options.maxContextTokens * 4;
    items = items.filter((r) => r.context.length <= maxChars);
  }
  // Deterministic order by id.
  items.sort((a, b) => a._id.localeCompare(b._id));
  if (options.limit !== undefined) items = items.slice(0, options.limit);
  return items;
}

export function buildPrompt(item: LongBenchItem): {
  query: string;
  context: string;
} {
  const query =
    `${item.question}\n\n` +
    `A) ${item.choice_A}\n` +
    `B) ${item.choice_B}\n` +
    `C) ${item.choice_C}\n` +
    `D) ${item.choice_D}\n\n` +
    `Respond with a single letter: A, B, C, or D. Then on a new line, write "FINAL: <letter>".`;
  return { query, context: item.context };
}

/** Extract the model's answer letter. Accepts "A", "(A)", "FINAL: A", etc. */
export function scoreCodeQA(
  item: LongBenchItem,
  answer: string,
): { correct: boolean; predicted: string | null } {
  const text = answer.toUpperCase();
  // Prefer an explicit "FINAL: X" line.
  const finalMatch = text.match(/FINAL\s*:\s*([ABCD])/);
  let predicted: string | null = finalMatch ? finalMatch[1]! : null;
  if (!predicted) {
    // Fall back: look for a parenthesised or isolated letter.
    const paren = text.match(/\(([ABCD])\)/);
    if (paren) predicted = paren[1]!;
  }
  if (!predicted) {
    // Last resort: first isolated ABCD letter.
    const iso = text.match(/(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)/);
    if (iso) predicted = iso[1]!;
  }
  return { correct: predicted === item.answer, predicted };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const items = loadCodeQA({ maxContextTokens: 128_000, limit: 10 });
  console.log(`loaded ${items.length} items`);
  for (const it of items) {
    console.log(
      `  ${it._id} len=${Math.ceil(it.context.length / 4)}tk diff=${it.difficulty} answer=${it.answer}`,
    );
  }
}
