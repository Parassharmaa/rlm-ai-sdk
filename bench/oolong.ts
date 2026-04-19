/**
 * OOLONG-synth loader (counting tasks at 32K tokens).
 *
 * Dataset: https://huggingface.co/datasets/oolongbench/oolong-synth
 * Paper:   https://arxiv.org/abs/2511.02817
 *
 * We pull the `counting` task_group at ctx_len=32768 — a pure label-aggregation
 * task where the model must classify N items and count. This stresses the
 * recursive sub-call path: map over chunks, count per chunk, reduce.
 *
 * Fields we care about: id, task (MOST_FREQ / LEAST_FREQ / RELATIVE_FREQ),
 * context_len, context_window_text, question, answer (stringified list).
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

export interface OolongItem {
  id: number;
  task: string;
  context_len: number;
  num_labels: number;
  context_window_text: string;
  question: string;
  answer: string[];
}

const CACHE_PATH = join("bench", "data", "oolong-count-32k.json");

/**
 * Fetch N counting@32K items from the HF datasets-server REST API.
 * Cached to disk so subsequent runs are offline.
 */
export async function fetchOolongCount32K(options: {
  n: number;
  offset?: number;
}): Promise<OolongItem[]> {
  if (existsSync(CACHE_PATH)) {
    const cached = JSON.parse(readFileSync(CACHE_PATH, "utf8")) as OolongItem[];
    if (cached.length >= options.n) {
      return cached.slice(options.offset ?? 0, (options.offset ?? 0) + options.n);
    }
  }
  mkdirSync(join("bench", "data"), { recursive: true });

  // Pull in pages of 100 from the test split and filter client-side.
  const collected: OolongItem[] = [];
  const wanted = (options.offset ?? 0) + options.n;
  const target = Math.max(wanted, 30);
  let offset = 0;
  const PAGE = 50;
  const MAX_PAGES = 80;
  const fetchPage = async (off: number): Promise<{ rows: Array<{ row: OolongItem }>; num_rows_total?: number }> => {
    const url =
      `https://datasets-server.huggingface.co/rows?` +
      `dataset=oolongbench/oolong-synth&config=default&split=test` +
      `&offset=${off}&length=${PAGE}`;
    for (let attempt = 0; attempt < 5; attempt++) {
      const res = await fetch(url);
      if (res.ok) return (await res.json()) as {
        rows: Array<{ row: OolongItem }>;
        num_rows_total?: number;
      };
      if (res.status === 500 || res.status === 503 || res.status === 429) {
        const wait = res.status === 429 ? 2_500 * (attempt + 1) : 500 * (attempt + 1);
        await new Promise((r) => setTimeout(r, wait));
        continue;
      }
      // Non-transient error: skip this page and move on.
      return { rows: [] };
    }
    // Retries exhausted — skip.
    return { rows: [] };
  };
  for (let p = 0; p < MAX_PAGES && collected.length < target; p++) {
    const data = await fetchPage(offset);
    if (!data.rows?.length) break;
    // Small pause to be polite and avoid 429.
    await new Promise((r) => setTimeout(r, 300));
    for (const r of data.rows) {
      if (r.row.task_group === undefined) {
        // no task_group filter available; filter on task prefix and ctx_len
      }
      if (
        r.row.context_len === 32768 &&
        (r.row.task === "TASK_TYPE.MOST_FREQ" ||
          r.row.task === "TASK_TYPE.LEAST_FREQ")
      ) {
        // Normalise answer to an array of strings.
        const rawAns = r.row.answer as unknown;
        const ans =
          typeof rawAns === "string"
            ? parsePyList(rawAns)
            : Array.isArray(rawAns)
              ? rawAns.map(String)
              : [String(rawAns)];
        // Skip items whose answer isn't a discrete label (numeric user-id
        // tasks sneak in under the same task types).
        const labels = labelsFromQuestion(r.row.question);
        const goldStr = String(ans[0] ?? "");
        if (!goldStr || /^-?\d+$/.test(goldStr)) continue;
        // Require parsed labels and that the gold is one of them.
        if (labels.length === 0) continue;
        if (!labels.some((l) => l.toLowerCase() === goldStr.toLowerCase())) continue;
        collected.push({ ...(r.row as OolongItem), answer: ans });
      }
    }
    offset += PAGE;
    if (data.num_rows_total !== undefined && offset >= data.num_rows_total) break;
  }
  writeFileSync(CACHE_PATH, JSON.stringify(collected, null, 2));
  const start = options.offset ?? 0;
  return collected.slice(start, start + options.n);
}

export function buildPrompt(item: OolongItem): {
  query: string;
  context: string;
} {
  const query = `${item.question}\n\nRespond with exactly one line in the form: "Label: <answer>".`;
  return { query, context: item.context_window_text };
}

/**
 * Extract the predicted label from free-text answer.
 * Accepts "Label: X", "the answer is X", or a bare label word.
 */
export function extractLabel(
  candidateLabels: string[],
  answer: string,
): string | null {
  const lower = answer.toLowerCase();
  // Prefer explicit "Label: X".
  const m = lower.match(/label\s*:\s*['"]?([a-z0-9_\- ]+?)['"]?\s*(?:$|[.!\n])/);
  if (m) {
    const candidate = m[1]!.trim();
    for (const l of candidateLabels) {
      if (candidate === l.toLowerCase()) return l;
    }
  }
  // Scan for any label mention; pick the LAST one (likely the conclusion).
  let last: string | null = null;
  for (const l of candidateLabels) {
    const re = new RegExp(`\\b${l.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}\\b`, "gi");
    if (re.test(answer)) last = l;
  }
  return last;
}

export function scoreOolong(
  item: OolongItem,
  candidateLabels: string[],
  answer: string,
): { correct: boolean; predicted: string | null; gold: string } {
  const gold = Array.isArray(item.answer) ? item.answer[0]! : item.answer;
  const predicted = extractLabel(candidateLabels, answer);
  // For RELATIVE_FREQ tasks, gold is a phrase like "less common than" — the
  // scorer above is specific to labels, so for those we fall back to substring.
  const relative = item.task === "TASK_TYPE.RELATIVE_FREQ";
  if (relative) {
    const matched = answer.toLowerCase().includes(gold.toLowerCase());
    return { correct: matched, predicted: matched ? gold : null, gold };
  }
  return { correct: predicted === gold, predicted, gold };
}

/** Parse candidate labels from the question ("one of the labels: a, b, c"). */
export function labelsFromQuestion(question: string): string[] {
  // Look for the LAST occurrence of "one of the labels:" so we don't catch
  // "the form 'Label: answer'" earlier in the question.
  const re = /one of the labels?:\s*([^.\n]+?)(?:\.|$)/gi;
  let match: RegExpExecArray | null;
  let best: string | null = null;
  while ((match = re.exec(question)) !== null) best = match[1]!;
  if (!best) return [];
  return best
    .split(/,|\bor\b|\band\b/)
    .map((s) => s.trim().replace(/^['"]|['"]$/g, ""))
    .filter(Boolean);
}

/** Parse a Python-style list string `"['a', 'b']"` → `["a", "b"]`. */
function parsePyList(s: string): string[] {
  const trimmed = s.trim();
  if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) return [trimmed];
  try {
    // Swap Python quotes for JSON.
    return JSON.parse(trimmed.replace(/'/g, '"')) as string[];
  } catch {
    return [trimmed.slice(1, -1)];
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const items = await fetchOolongCount32K({ n: 15 });
  console.log(`cached ${items.length} items`);
  for (const it of items) {
    console.log(
      `  ${it.id} task=${it.task} ctx=${it.context_len} labels=${it.num_labels} A=${JSON.stringify(it.answer)}`,
    );
    const labels = labelsFromQuestion(it.question);
    console.log(`    parsed labels: ${JSON.stringify(labels)}`);
  }
}
