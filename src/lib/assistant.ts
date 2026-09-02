/**
 * AI Registration Assistant — NLP core (TypeScript port of the Python engine).
 * Task AI-SS-001 · Student Code DAS008940
 *
 * Pipeline: preprocess -> intent classification (bag-of-words cosine scoring)
 *           -> entity extraction (regex) -> validation -> dialog state machine.
 */

// ---------------------------------------------------------------- preprocessing

const STOP_WORDS = new Set([
  "a","an","the","is","am","are","was","were","be","been","being","i","me","my","you","your",
  "he","she","it","we","they","of","to","for","in","on","at","and","or","but","with","as",
  "do","does","did","so","that","this","there","here","have","has","had","will","would","can",
]);

const SUFFIXES = ["ing", "edly", "ies", "es", "ed", "ly", "s"];

/** Naive lemmatizer: strips common English suffixes (stand-in for WordNetLemmatizer). */
export function lemmatize(token: string): string {
  for (const suf of SUFFIXES) {
    if (token.length > suf.length + 2 && token.endsWith(suf)) {
      const base = token.slice(0, -suf.length);
      return suf === "ies" ? `${base}y` : base;
    }
  }
  return token;
}

export function preprocess(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t && !STOP_WORDS.has(t))
    .map(lemmatize);
}

// ---------------------------------------------------------------- intents

export type IntentName =
  | "greeting"
  | "register"
  | "help"
  | "faq_duration"
  | "faq_fee"
  | "faq_certificate"
  | "faq_skills"
  | "status"
  | "restart"
  | "thank_you"
  | "goodbye"
  | "unknown";

interface Intent {
  patterns: string[];
  responses: string[];
}

export const INTENTS: Record<Exclude<IntentName, "unknown">, Intent> = {
  greeting: {
    patterns: ["hi", "hello", "hey", "good morning", "good evening", "namaste", "greetings"],
    responses: [
      "Hello! Welcome to the Free Online AI & Data Science Internship registration. Type **register** whenever you're ready, or ask me anything.",
    ],
  },
  register: {
    patterns: ["register", "registration", "apply", "sign up", "signup", "join", "enroll", "start"],
    responses: ["Great! Let's get you registered. First — what is your full name?"],
  },
  help: {
    patterns: ["help", "support", "assist", "guide", "what can you do", "options"],
    responses: [
      "I can help you with:\n- Starting your **registration**\n- Internship **duration**, **fee** and **certificate**\n- Required **skills**\n- Your registration **status**\n\nType **register** to begin.",
    ],
  },
  faq_duration: {
    patterns: ["duration", "how long", "weeks", "timeline", "period", "days"],
    responses: [
      "The internship task runs for **7 days**, roughly 3 hours a day — from research and planning on Day 1 to submission on Day 7.",
    ],
  },
  faq_fee: {
    patterns: ["fee", "cost", "price", "free", "payment", "charge", "money"],
    responses: ["The internship is completely **free** — there is no registration or certification fee."],
  },
  faq_certificate: {
    patterns: ["certificate", "certification", "letter", "recommendation", "proof"],
    responses: [
      "Yes — you receive a **completion certificate** after submitting your GitHub repository, demo video and blog post.",
    ],
  },
  faq_skills: {
    patterns: ["skill", "requirement", "prerequisite", "eligibility", "need to know", "python"],
    responses: [
      "Basic **Python** is enough to start. You'll pick up NLP (NLTK/spaCy), intent classification and entity extraction along the way.",
    ],
  },
  status: {
    patterns: ["status", "my data", "progress", "summary", "what did i give", "details"],
    responses: [""],
  },
  restart: {
    patterns: ["restart", "reset", "start over", "clear", "again from beginning"],
    responses: ["No problem — I've cleared everything. Type **register** to start again."],
  },
  thank_you: {
    patterns: ["thank", "thanks", "appreciate", "grateful"],
    responses: ["You're welcome! Anything else I can help you with?"],
  },
  goodbye: {
    patterns: ["bye", "goodbye", "exit", "quit", "see you"],
    responses: ["Thank you for using the AI Registration Assistant. Good luck with the internship!"],
  },
};

/** Cosine-style similarity between the message tokens and each intent's pattern bag. */
export function classifyIntent(text: string): { intent: IntentName; confidence: number } {
  const lower = text.toLowerCase();
  const tokens = preprocess(text);
  let best: IntentName = "unknown";
  let bestScore = 0;

  for (const [name, data] of Object.entries(INTENTS)) {
    let score = 0;
    for (const pattern of data.patterns) {
      if (pattern.includes(" ")) {
        if (lower.includes(pattern)) score += 1;
        continue;
      }
      const lemma = lemmatize(pattern);
      if (tokens.includes(lemma)) score += 1;
      else if (new RegExp(`\\b${pattern}\\b`).test(lower)) score += 0.8;
    }
    const normalized = score / Math.sqrt(data.patterns.length);
    if (normalized > bestScore) {
      bestScore = normalized;
      best = name as IntentName;
    }
  }

  return bestScore >= 0.3 ? { intent: best, confidence: Math.min(1, bestScore) } : { intent: "unknown", confidence: bestScore };
}

// ---------------------------------------------------------------- entities

export interface Entities {
  name?: string;
  email?: string;
  phone?: string;
  field?: string;
  experience?: string;
}

const FIELDS = [
  "computer science","information technology","data science","artificial intelligence",
  "electronics","mechanical","civil","electrical","mathematics","statistics","physics",
  "commerce","business","engineering","bca","mca","bsc","msc","btech","mtech",
];

const EXPERIENCE = [
  { level: "Beginner", words: ["beginner", "fresher", "new", "none", "zero", "basic", "started"] },
  { level: "Intermediate", words: ["intermediate", "some", "moderate", "average", "one year", "medium"] },
  { level: "Advanced", words: ["advanced", "expert", "experienced", "professional", "years", "senior"] },
];

export function extractEntities(text: string): Entities {
  const e: Entities = {};

  const nameMatch = text.match(/(?:my name is|i am|i'm|this is|name[:\s-]+)\s*([a-zA-Z][a-zA-Z.\s]{1,49})/i);
  if (nameMatch?.[1]) e.name = titleCase(nameMatch[1]);

  const emailMatch = text.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/);
  if (emailMatch) e.email = emailMatch[0].toLowerCase();

  const phoneMatch = text.match(/(?:\+\d{1,3}[\s-]?)?\d{10}\b/);
  if (phoneMatch) e.phone = phoneMatch[0].replace(/\s|-/g, "");

  const lower = text.toLowerCase();
  const field = FIELDS.find((f) => lower.includes(f));
  if (field) e.field = titleCase(field);

  const exp = EXPERIENCE.find((x) => x.words.some((w) => lower.includes(w)));
  if (exp) e.experience = exp.level;

  return e;
}

function titleCase(s: string): string {
  return s
    .trim()
    .replace(/\s+/g, " ")
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

// ---------------------------------------------------------------- validation

export const validators = {
  name: (v: string) =>
    /^[a-zA-Z][a-zA-Z.\s]{2,49}$/.test(v.trim()) && v.trim().split(/\s+/).length >= 2
      ? null
      : "That doesn't look like a full name. Please enter your **first and last name** (letters only).",
  email: (v: string) =>
    /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/.test(v.trim())
      ? null
      : "That email looks invalid. Please use a format like **you@example.com**.",
  phone: (v: string) =>
    /^(\+\d{1,3})?\d{10}$/.test(v.replace(/\s|-/g, ""))
      ? null
      : "Please enter a valid **10-digit phone number**.",
  field: (v: string) => (v.trim().length >= 2 ? null : "Please tell me your field of study."),
  experience: (v: string) => (v.trim().length >= 2 ? null : "Please describe your programming experience."),
};

// ---------------------------------------------------------------- sentiment

const POSITIVE = ["good","great","awesome","excited","happy","love","nice","thanks","thank","excellent","perfect","cool","yes"];
const NEGATIVE = ["bad","confused","stuck","angry","hate","worst","difficult","hard","not working","frustrated","sad","problem","issue"];

export type Sentiment = "positive" | "neutral" | "negative";

export function analyzeSentiment(text: string): Sentiment {
  const lower = text.toLowerCase();
  const pos = POSITIVE.filter((w) => lower.includes(w)).length;
  const neg = NEGATIVE.filter((w) => lower.includes(w)).length;
  if (pos > neg) return "positive";
  if (neg > pos) return "negative";
  return "neutral";
}

// ---------------------------------------------------------------- dialog manager

export type Slot = "name" | "email" | "phone" | "field" | "experience";
export const SLOT_ORDER: Slot[] = ["name", "email", "phone", "field", "experience"];

const PROMPTS: Record<Slot, string> = {
  name: "What is your **full name**?",
  email: "Thanks! What's your **email address**?",
  phone: "Got it. What's your **10-digit phone number**?",
  field: "Which **field of study** are you in? (e.g. Computer Science)",
  experience: "Last one — describe your **programming experience** (beginner / intermediate / advanced).",
};

export interface Registration {
  id: string;
  name: string;
  email: string;
  phone: string;
  field: string;
  experience: string;
  registeredAt: string;
  taskId: "AI-SS-001";
  studentCode: "DAS008940";
}

export interface AssistantState {
  collecting: boolean;
  data: Partial<Record<Slot, string>>;
  completed: Registration | null;
}

export const initialState: AssistantState = { collecting: false, data: {}, completed: null };

export interface Turn {
  reply: string;
  state: AssistantState;
  intent: IntentName;
  confidence: number;
  entities: Entities;
  sentiment: Sentiment;
}

function nextSlot(data: AssistantState["data"]): Slot | null {
  return SLOT_ORDER.find((s) => !data[s]) ?? null;
}

function confirmation(data: Required<Record<Slot, string>>): Registration {
  return {
    id: `REG-${Math.random().toString(36).slice(2, 7).toUpperCase()}`,
    name: data.name,
    email: data.email,
    phone: data.phone,
    field: data.field,
    experience: data.experience,
    registeredAt: new Date().toISOString(),
    taskId: "AI-SS-001",
    studentCode: "DAS008940",
  };
}

/** Single conversation turn: intent + entities + slot filling + validation. */
export function respond(input: string, state: AssistantState): Turn {
  const text = input.trim();
  const { intent, confidence } = classifyIntent(text);
  const entities = extractEntities(text);
  const sentiment = analyzeSentiment(text);
  const base = { intent, confidence, entities, sentiment };

  if (intent === "restart") {
    return { ...base, reply: INTENTS.restart.responses[0]!, state: { ...initialState } };
  }

  if (intent === "status") {
    return { ...base, reply: statusReport(state), state };
  }

  // Global FAQ / small-talk handling, even mid-registration.
  const faqIntents: IntentName[] = ["help", "faq_duration", "faq_fee", "faq_certificate", "faq_skills", "goodbye", "thank_you"];
  if (faqIntents.includes(intent) && confidence >= 0.5) {
    const answer = INTENTS[intent as keyof typeof INTENTS].responses[0];
    const pending = state.collecting ? `\n\nBack to it — ${PROMPTS[nextSlot(state.data) ?? "name"]}` : "";
    return { ...base, reply: answer + pending, state };
  }

  if (!state.collecting) {
    if (intent === "register" || Object.keys(entities).length > 0) {
      const data = { ...state.data };
      applyEntities(data, entities);
      const slot = nextSlot(data);
      const newState = { ...state, collecting: true, data, completed: null };
      const greet = intent === "register" ? "Great! Let's get you registered.\n\n" : "Let's get you registered.\n\n";
      if (!slot) return finish(base, newState);
      return { ...base, reply: greet + PROMPTS[slot], state: newState };
    }
    if (intent === "greeting") {
      return { ...base, reply: INTENTS.greeting.responses[0]!, state };
    }
    return {
      ...base,
      reply:
        "I'm not sure I understood that. You can type **register** to begin, **help** to see what I can do, or ask about the duration, fee or certificate.",
      state,
    };
  }

  // --- collecting mode: fill the current slot -------------------------------
  const slot = nextSlot(state.data)!;
  const candidate = entities[slot] ?? rawCandidate(slot, text);
  const error = validators[slot](candidate ?? "");

  if (error) {
    const empathy = sentiment === "negative" ? "No worries, we'll sort this out. " : "";
    return { ...base, reply: empathy + error, state };
  }

  const data = { ...state.data, [slot]: normalize(slot, candidate!) };
  applyEntities(data, entities);
  const newState = { ...state, data };
  const upcoming = nextSlot(data);

  if (!upcoming) return finish(base, newState);

  const ack = slot === "name" ? `Nice to meet you, **${data.name}**! ` : "Saved. ";
  return { ...base, reply: ack + PROMPTS[upcoming], state: newState };
}

function rawCandidate(slot: Slot, text: string): string {
  if (slot === "field" || slot === "experience" || slot === "name") return text.replace(/[^a-zA-Z.\s]/g, " ").trim();
  return text.trim();
}

function normalize(slot: Slot, value: string): string {
  if (slot === "email") return value.toLowerCase();
  if (slot === "phone") return value.replace(/\s|-/g, "");
  if (slot === "experience") {
    return extractEntities(value).experience ?? titleCase(value);
  }
  return titleCase(value);
}

function applyEntities(data: AssistantState["data"], entities: Entities) {
  for (const slot of SLOT_ORDER) {
    const v = entities[slot];
    if (v && !data[slot] && !validators[slot](v)) data[slot] = normalize(slot, v);
  }
}

function finish(base: Omit<Turn, "reply" | "state">, state: AssistantState): Turn {
  const registration = confirmation(state.data as Required<Record<Slot, string>>);
  const reply = [
    "🎉 **Registration confirmed!**",
    "",
    `**ID:** ${registration.id}`,
    `**Name:** ${registration.name}`,
    `**Email:** ${registration.email}`,
    `**Phone:** ${registration.phone}`,
    `**Field:** ${registration.field}`,
    `**Experience:** ${registration.experience}`,
    "",
    "A copy of your details is saved as JSON on the right. Type **restart** to register someone else.",
  ].join("\n");
  return { ...base, reply, state: { collecting: false, data: state.data, completed: registration } };
}

function statusReport(state: AssistantState): string {
  const filled = SLOT_ORDER.filter((s) => state.data[s]);
  if (!filled.length) return "I haven't collected anything yet. Type **register** to begin.";
  const lines = filled.map((s) => `- **${s}:** ${state.data[s]}`);
  const missing = SLOT_ORDER.filter((s) => !state.data[s]);
  const tail = missing.length ? `\n\nStill needed: ${missing.join(", ")}.` : "\n\nAll details collected.";
  return `Here's what I have so far:\n${lines.join("\n")}${tail}`;
}
