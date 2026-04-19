/**
 * Synthetic pairs task — quadratic aggregation designed to favor the
 * RLM sub-call path.
 *
 * Each item has ~40 "users", each with a favorite-food and a least-favorite-
 * food. Filler text per user pads total context to ~32K tokens. The query:
 *
 *   How many pairs (A,B) with A ≠ B are there such that A's favorite food
 *   equals B's least favorite food?
 *
 * This is genuinely quadratic: for each user, the model must know their two
 * attributes, and then iterate over all ~40² / 2 pairs. That structure is
 * where sub-calls should help:
 *   - Sub-calls per-user extract attributes into a compact table.
 *   - Root LM does the pairwise comparison over the table (cheap).
 *
 * Bash-only RLM can grep for "favorite" vs "least favorite" but still has
 * to do the classification+matching itself in its attention.
 *
 * Deterministic via seeded PRNG.
 */

const FOODS = [
  "pizza", "sushi", "ramen", "tacos", "burgers", "pasta", "salad", "curry",
  "falafel", "dumplings", "bbq", "steak", "pho", "burritos", "paella",
  "soup", "pancakes", "waffles", "cheesecake", "kebab",
];

const FILLER_SENTENCES = [
  "they usually eat at home on weeknights",
  "they keep a garden in summer and grow herbs",
  "they travel for work about once a quarter",
  "their kitchen is always tidy by Sunday evening",
  "they have strong opinions about olive oil brands",
  "they avoid caffeine after four in the afternoon",
  "they host a dinner party most months",
  "they read food magazines on airplanes",
  "they moved to the city six years ago",
  "their favorite day of the week is Saturday",
  "they like to cook while listening to podcasts",
  "they have adopted two cats named after spices",
  "they run a small blog nobody reads",
  "they ski every February without fail",
  "they learnt to bake bread during a recent winter",
];

function prng(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface PairsUser {
  id: number;
  name: string;
  favorite: string;
  leastFavorite: string;
  blurb: string;
}

export interface PairsItem {
  id: string;
  users: PairsUser[];
  context: string;
  question: string;
  answer: number;
}

/** Generate a deterministic pairs item. */
export function generatePairsItem(
  seed: number,
  numUsers = 40,
  fillerSentencesPerUser = 20,
): PairsItem {
  const rng = prng(seed);
  const pick = <T>(arr: T[]): T => arr[Math.floor(rng() * arr.length)]!;

  const users: PairsUser[] = [];
  for (let i = 0; i < numUsers; i++) {
    // Favorite and least-favorite must differ.
    let fav = pick(FOODS);
    let least = pick(FOODS);
    while (least === fav) least = pick(FOODS);

    const blurbs: string[] = [];
    for (let k = 0; k < fillerSentencesPerUser; k++) {
      blurbs.push(pick(FILLER_SENTENCES));
    }

    users.push({
      id: 1000 + i,
      name: `User #${1000 + i}`,
      favorite: fav,
      leastFavorite: least,
      blurb: blurbs.join(". ") + ".",
    });
  }

  // Count pairs (A, B) with A ≠ B where A.favorite == B.leastFavorite.
  // Ordered pairs — (A,B) and (B,A) count separately because "A's favorite
  // equals B's least" is not symmetric with "B's favorite equals A's least".
  let answer = 0;
  for (const a of users) {
    for (const b of users) {
      if (a.id === b.id) continue;
      if (a.favorite === b.leastFavorite) answer++;
    }
  }

  // Compose the context: each user as a block; shuffle block order.
  const blocks: string[] = [];
  for (const u of users) {
    blocks.push(
      [
        `--- ${u.name} ---`,
        `Favorite food: ${u.favorite}.`,
        `Least favorite food: ${u.leastFavorite}.`,
        `Notes: ${u.blurb}`,
      ].join("\n"),
    );
  }
  // Shuffle block order so position doesn't leak info.
  for (let i = blocks.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [blocks[i], blocks[j]] = [blocks[j]!, blocks[i]!];
  }

  const preamble =
    `The following directory contains ${numUsers} user profiles, one per block, ` +
    `separated by "--- User #NNNN ---" headers. Each profile lists the user's ` +
    `favorite food and least favorite food plus some notes.`;

  const context = `${preamble}\n\n${blocks.join("\n\n")}`;

  const question =
    `How many ORDERED pairs (A, B) with A ≠ B are there in the profiles ` +
    `such that A's favorite food equals B's least favorite food? ` +
    `Respond with exactly one line in the form: "COUNT: <number>".`;

  return {
    id: `pairs-${seed}`,
    users,
    context,
    question,
    answer,
  };
}

export function scorePairs(
  item: PairsItem,
  answer: string,
): { correct: boolean; predicted: number | null; gold: number } {
  const m = answer.match(/COUNT\s*:\s*(-?\d+)/i);
  if (m) {
    const predicted = parseInt(m[1]!, 10);
    return { correct: predicted === item.answer, predicted, gold: item.answer };
  }
  // Fall back to any bare integer.
  const any = answer.match(/\b(\d{1,4})\b/);
  if (any) {
    const predicted = parseInt(any[1]!, 10);
    return { correct: predicted === item.answer, predicted, gold: item.answer };
  }
  return { correct: false, predicted: null, gold: item.answer };
}

export function buildPrompt(item: PairsItem): {
  query: string;
  context: string;
} {
  return { query: item.question, context: item.context };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  for (let i = 0; i < 3; i++) {
    const item = generatePairsItem(100 + i);
    const tokens = Math.ceil(item.context.length / 4);
    console.log(
      `${item.id}: ${item.users.length} users, ${item.context.length} chars (~${tokens} tokens), answer=${item.answer}`,
    );
  }
}
