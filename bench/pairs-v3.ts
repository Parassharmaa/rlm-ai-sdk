/**
 * Pairs V3 — attributes embedded in natural-language paragraphs instead of
 * labelled fields. No "Favorite food: X" structure; the model has to read
 * each user's story to figure out what they love and what they hate.
 *
 * Designed to make bash text-processing hard (varied phrasings, food names
 * also mentioned neutrally in filler) and therefore force the RLM to use
 * sub-calls for per-user attribute extraction.
 *
 * Task: same as pairs.ts — count ordered (A, B) where A.favorite ==
 * B.least_favorite.
 */

const FOODS = [
  "pizza", "sushi", "ramen", "tacos", "burgers", "pasta", "salad", "curry",
  "falafel", "dumplings", "bbq", "steak", "pho", "burritos", "paella",
  "soup", "pancakes", "waffles", "cheesecake", "kebab",
];

const NAMES = [
  "Alice", "Blake", "Casey", "Devon", "Emery", "Finley", "Grey", "Harper",
  "Indigo", "Jordan", "Kai", "Lane", "Morgan", "Nova", "Oakley", "Parker",
  "Quinn", "Riley", "Sage", "Tate", "Umi", "Val", "Wren", "Xander",
  "Yael", "Zion", "Ash", "Bailey", "Coral", "Dex", "Ember", "Fox",
  "Glenn", "Hollis", "Iris", "Jun", "Kit", "Leighton", "Mica", "Noor",
];

/** Favorite-describing templates. The food appears inside a positive
 *  sentiment frame but the word "favorite" never appears. */
const FAV_TEMPLATES = [
  "Growing up in a small coastal town, {name}'s summers were defined by their family's Sunday tradition of gathering at the pier for {fav}. Those are the happiest memories they have, and decades later {name} will still go out of their way to find a good version of it whenever they travel.",
  "{name} first tried {fav} on a backpacking trip in their twenties and it changed everything — they spent the next year learning to cook it from scratch, and now refuse to date anyone who doesn't appreciate it. Friends know to always pick the {fav} place when {name}'s birthday comes around.",
  "When {name} had surgery last year, the one thing that got them through recovery was their partner bringing home {fav} from the little place around the corner. Even now, every good day seems to end with them ordering another bowl.",
  "There's a framed photo in {name}'s kitchen of the very first {fav} they made themselves — they've been tweaking the recipe for eight years and it's basically a household religion at this point. Nothing makes a bad day better than a fresh batch.",
  "{name} runs a small blog about {fav} — they've reviewed every place serving it within a thirty-mile radius and argue passionately about which cuisine does it best. It's the only food they can talk about for an hour without getting bored.",
];

/** Least-favorite templates. */
const LEAST_TEMPLATES = [
  "{name} has a story about {least} from a bad family reunion in 2019 that they'll tell whenever it comes up — to this day they'll politely decline it at every dinner party, citing \"personal history.\" Their partner has given up on getting them to try it again.",
  "{name} won't even walk into a restaurant that has {least} as the house specialty — they find the smell unbearable and have been known to sit outside when friends insist on eating there. It's been a lifelong thing, not a phase.",
  "Once at a team lunch {name} ordered {least} without realizing what it was, and ended up eating granola bars from the vending machine instead. They still bring it up as one of the worst meals of their life.",
  "Every time {least} comes up in conversation, {name} makes that face — the one their sister has been imitating for fifteen years. They can't explain it rationally, but a single bite ruins an entire evening for them.",
  "{name}'s running joke with their college roommates was the \"{least} contract\" — a pledge never to serve it in their apartment. They take it more seriously than the joke suggests and have been honoring it for a decade.",
];

/** Neutral filler that mentions random foods (NOT the user's fav or least).
 *  These make grep-for-food-name strategies fail because the food appears
 *  in neutral contexts too. */
const FILLER_TEMPLATES = [
  "On weekdays {name} usually packs something simple — a sandwich, sometimes {filler}, whatever's quick.",
  "Their apartment is near a row of food trucks that all do different things; the {filler} one is fine but nothing special.",
  "When colleagues order group lunch they'll shrug and go along with whatever — {filler} today, something else tomorrow.",
  "{name}'s cousin runs a catering business that does {filler} for weddings; {name} helps out twice a year.",
  "There's a long-running group text where friends rate new restaurants — last month they covered a {filler} place that was just okay.",
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

export interface PairsV3User {
  name: string;
  favorite: string;
  leastFavorite: string;
  paragraph: string;
}

export interface PairsV3Item {
  id: string;
  users: PairsV3User[];
  context: string;
  question: string;
  answer: number;
}

export function generatePairsV3Item(
  seed: number,
  numUsers = 40,
): PairsV3Item {
  const rng = prng(seed);
  const pick = <T>(arr: T[]): T => arr[Math.floor(rng() * arr.length)]!;
  const shuffle = <T>(arr: T[]): T[] => {
    const out = [...arr];
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [out[i], out[j]] = [out[j]!, out[i]!];
    }
    return out;
  };

  const shuffledNames = shuffle(NAMES).slice(0, numUsers);

  const users: PairsV3User[] = [];
  for (let i = 0; i < numUsers; i++) {
    let fav = pick(FOODS);
    let least = pick(FOODS);
    while (least === fav) least = pick(FOODS);
    const name = shuffledNames[i] ?? `User${i}`;

    // Sentences order — intersperse fav, least, and 2-4 neutral mentions.
    const sentences: string[] = [];
    const favTmpl = pick(FAV_TEMPLATES);
    const leastTmpl = pick(LEAST_TEMPLATES);
    sentences.push(favTmpl.replaceAll("{name}", name).replaceAll("{fav}", fav));
    sentences.push(leastTmpl.replaceAll("{name}", name).replaceAll("{least}", least));
    const numFillers = 2 + Math.floor(rng() * 3);
    for (let k = 0; k < numFillers; k++) {
      let filler = pick(FOODS);
      // Filler food should not be this user's fav or least so we don't
      // accidentally teach the task "last mention wins".
      while (filler === fav || filler === least) filler = pick(FOODS);
      const tmpl = pick(FILLER_TEMPLATES);
      sentences.push(
        tmpl.replaceAll("{name}", name).replaceAll("{filler}", filler),
      );
    }
    // Shuffle sentence order so fav and least aren't always in the same slot.
    const shuffled = shuffle(sentences);
    users.push({
      name,
      favorite: fav,
      leastFavorite: least,
      paragraph: shuffled.join(" "),
    });
  }

  // Count ordered pairs.
  let answer = 0;
  for (const a of users) {
    for (const b of users) {
      if (a === b) continue;
      if (a.favorite === b.leastFavorite) answer++;
    }
  }

  // Assemble context. Randomised block order.
  const blocks = shuffle(
    users.map((u) => `### Profile of ${u.name}\n\n${u.paragraph}`),
  );

  const preamble =
    `The following directory contains ${numUsers} personal profiles. ` +
    `Each profile is a short free-form paragraph about one person's food ` +
    `preferences and habits — written in natural language, with no ` +
    `explicit "favorite:" or "least-favorite:" labels. You must read each ` +
    `profile to determine what food each person loves most and what food ` +
    `they cannot stand. Foods mentioned in passing (what they ate on ` +
    `Tuesday, what their cousin cooks, etc.) are neutral and should not ` +
    `count as favorites or least-favorites.`;

  const context = `${preamble}\n\n${blocks.join("\n\n---\n\n")}`;

  const question =
    `How many ORDERED pairs (A, B) with A ≠ B are there in the profiles ` +
    `such that A's single favorite food equals B's single least favorite ` +
    `food? (Each person has exactly one clear favorite and exactly one ` +
    `clear food they cannot stand, with the rest mentioned only neutrally.) ` +
    `Respond with exactly one line in the form: "COUNT: <number>".`;

  return {
    id: `pairs-v3-${seed}`,
    users,
    context,
    question,
    answer,
  };
}

export function scorePairsV3(
  item: PairsV3Item,
  answer: string,
): { correct: boolean; predicted: number | null; gold: number } {
  const m = answer.match(/COUNT\s*:\s*(-?\d+)/i);
  if (m) {
    const predicted = parseInt(m[1]!, 10);
    return { correct: predicted === item.answer, predicted, gold: item.answer };
  }
  const any = answer.match(/\b(\d{1,4})\b/);
  if (any) {
    const predicted = parseInt(any[1]!, 10);
    return { correct: predicted === item.answer, predicted, gold: item.answer };
  }
  return { correct: false, predicted: null, gold: item.answer };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  for (let i = 0; i < 3; i++) {
    const item = generatePairsV3Item(2_000_000 + i);
    const tokens = Math.ceil(item.context.length / 4);
    console.log(
      `${item.id}: ${item.users.length} users, ${item.context.length} chars (~${tokens} tokens), answer=${item.answer}`,
    );
    if (i === 0) {
      console.log("\n--- sample profile ---");
      console.log(item.users[0]?.paragraph);
      console.log(`(gold fav=${item.users[0]?.favorite}, least=${item.users[0]?.leastFavorite})`);
    }
  }
}
