/**
 * S-NIAH (single needle in a haystack). Models read a long passage with
 * one embedded "magic number" and must report it back. Paper uses
 * 2^13 .. 2^18 tokens.
 *
 * We generate lorem-ipsum-ish filler using a deterministic RNG so runs are
 * reproducible across conditions. Token estimates use chars/4.
 */
import { randomBytes } from "node:crypto";

const CORPUS_WORDS = [
  "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
  "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
  "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
  "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "aliquip",
  "ex", "ea", "commodo", "consequat", "duis", "aute", "irure", "in",
  "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat",
  "nulla", "pariatur", "excepteur", "sint", "occaecat", "cupidatat",
  "proident", "sunt", "culpa", "qui", "officia", "deserunt", "mollit",
  "anim", "id", "est", "laborum",
];

/** Mulberry32 PRNG, deterministic from a seed. */
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

export interface NIAHSample {
  id: string;
  lengthTokens: number; // target token count (chars/4)
  lengthChars: number; // actual chars of haystack
  needle: string; // the sentence actually embedded
  magicNumber: string; // gold answer
  haystack: string; // full passage
  question: string;
}

function randomNumber(rng: () => number): string {
  // 6-char alphanumeric, unambiguous (no 0/O/1/l).
  const alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
  let out = "";
  for (let i = 0; i < 6; i++) {
    out += alphabet[Math.floor(rng() * alphabet.length)];
  }
  return out;
}

function generateFiller(targetChars: number, rng: () => number): string {
  const lines: string[] = [];
  let chars = 0;
  let lineNum = 0;
  while (chars < targetChars) {
    const wordCount = 8 + Math.floor(rng() * 8);
    const words: string[] = [];
    for (let i = 0; i < wordCount; i++) {
      words.push(CORPUS_WORDS[Math.floor(rng() * CORPUS_WORDS.length)]);
    }
    const line = `Entry ${lineNum++}: ${words.join(" ")}.`;
    lines.push(line);
    chars += line.length + 1;
  }
  return lines.join("\n");
}

/** Generate a single S-NIAH sample at the target token count (chars/4). */
export function generateNIAH(
  targetTokens: number,
  seed: number,
): NIAHSample {
  const rng = prng(seed);
  const targetChars = targetTokens * 4;
  const filler = generateFiller(targetChars, rng);
  const magicNumber = randomNumber(rng);
  const needle = `The magic number is ${magicNumber}.`;

  // Insert the needle at a uniformly random position (line boundary).
  const lines = filler.split("\n");
  const insertAt = Math.floor(rng() * lines.length);
  lines.splice(insertAt, 0, needle);
  const haystack = lines.join("\n");

  return {
    id: `niah-${targetTokens}-${seed}`,
    lengthTokens: Math.ceil(haystack.length / 4),
    lengthChars: haystack.length,
    needle,
    magicNumber,
    haystack,
    question:
      "In the passage above, there is exactly one sentence of the form " +
      "\"The magic number is XXXXXX.\" — what is the magic number? " +
      "Respond with just the 6-character code and nothing else.",
  };
}

export function scoreNIAH(sample: NIAHSample, answer: string): boolean {
  // Case-insensitive substring match against the gold magic number.
  return answer.toUpperCase().includes(sample.magicNumber.toUpperCase());
}

/** Build a deterministic sweep: N samples at each length. */
export function buildNIAHSweep(
  lengths: number[],
  samplesPerLength: number,
): NIAHSample[] {
  const out: NIAHSample[] = [];
  for (const len of lengths) {
    for (let i = 0; i < samplesPerLength; i++) {
      // Seed encodes length+index so re-runs are identical.
      out.push(generateNIAH(len, len * 1000 + i));
    }
  }
  return out;
}

// Sanity check when run directly.
if (import.meta.url === `file://${process.argv[1]}`) {
  const s = generateNIAH(1000, 1);
  console.log(
    `sample: tokens=${s.lengthTokens} chars=${s.lengthChars} needle=${s.needle} magic=${s.magicNumber}`,
  );
  console.log(`first 200 chars: ${s.haystack.slice(0, 200)}...`);
  // Self-test: scorer should hit.
  console.log(`scorer(magic): ${scoreNIAH(s, s.magicNumber)}`);
  console.log(`scorer(wrong): ${scoreNIAH(s, "XXXXXX")}`);
  void randomBytes; // silence unused
}
