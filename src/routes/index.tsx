import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  respond,
  initialState,
  SLOT_ORDER,
  type AssistantState,
  type IntentName,
  type Sentiment,
  type Entities,
} from "@/lib/assistant";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AI Registration Assistant — Internship Chatbot (AI-SS-001)" },
      {
        name: "description",
        content:
          "Conversational AI assistant that guides students through internship registration with intent recognition, entity extraction and validation.",
      },
      { property: "og:title", content: "AI Registration Assistant — Internship Chatbot" },
      {
        property: "og:description",
        content:
          "NLP-powered chatbot: intent classification, entity extraction, validation and dialog management for internship registration.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Home,
});

interface Message {
  id: number;
  role: "bot" | "user";
  text: string;
  meta?: { intent: IntentName; confidence: number; sentiment: Sentiment; entities: Entities };
}

const WELCOME =
  "Hello! I'm the **AI Registration Assistant** for the Free Online AI & Data Science Internship.\n\nType **register** to begin, or ask me about the duration, fee, certificate or required skills.";

function Home() {
  const [messages, setMessages] = useState<Message[]>([{ id: 0, role: "bot", text: WELCOME }]);
  const [state, setState] = useState<AssistantState>(initialState);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, typing]);

  const send = (raw: string) => {
    const text = raw.trim();
    if (!text || typing) return;
    const turn = respond(text, state);
    setMessages((m) => [
      ...m,
      { id: m.length, role: "user", text, meta: { intent: turn.intent, confidence: turn.confidence, sentiment: turn.sentiment, entities: turn.entities } },
    ]);
    setInput("");
    setTyping(true);
    window.setTimeout(() => {
      setMessages((m) => [...m, { id: m.length, role: "bot", text: turn.reply }]);
      setState(turn.state);
      setTyping(false);
    }, 420);
  };

  const lastMeta = useMemo(() => [...messages].reverse().find((m) => m.meta)?.meta, [messages]);
  const filled = SLOT_ORDER.filter((s) => state.data[s]).length;

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto grid max-w-6xl gap-6 px-4 py-8 lg:grid-cols-[1fr_20rem] lg:py-12">
        <header className="lg:col-span-2">
          <p className="font-mono text-xs uppercase tracking-[0.25em] text-accent">Task AI-SS-001 · DAS008940</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">
            AI Registration <span className="text-gradient">Assistant</span>
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            An NLP chatbot that recognises intent, extracts entities, validates input and manages the
            registration dialog — no database, data is kept as JSON.
          </p>
        </header>

        {/* Chat */}
        <section className="glass flex h-[34rem] flex-col overflow-hidden rounded-2xl">
          <div className="flex items-center gap-3 border-b border-border/60 px-5 py-3">
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-accent" />
            </span>
            <span className="text-sm font-medium">Registration Bot</span>
            <span className="ml-auto font-mono text-xs text-muted-foreground">online</span>
          </div>

          <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
            {messages.map((m) => (
              <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
                <div
                  className={
                    m.role === "user"
                      ? "max-w-[80%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground"
                      : "max-w-[85%] rounded-2xl rounded-bl-sm bg-secondary px-4 py-2.5 text-sm text-secondary-foreground"
                  }
                >
                  <Rich text={m.text} />
                </div>
              </div>
            ))}
            {typing && (
              <div className="flex justify-start">
                <div className="flex gap-1 rounded-2xl rounded-bl-sm bg-secondary px-4 py-3">
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted-foreground"
                      style={{ animationDelay: `${i * 120}ms` }}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-border/60 px-5 py-3">
            <div className="mb-2 flex flex-wrap gap-1.5">
              {["register", "help", "duration", "is it free?", "certificate", "status", "restart"].map((q) => (
                <button
                  key={q}
                  onClick={() => send(q)}
                  className="rounded-full border border-border px-2.5 py-1 font-mono text-[11px] text-muted-foreground transition-colors hover:border-accent hover:text-accent"
                >
                  {q}
                </button>
              ))}
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                send(input);
              }}
              className="flex gap-2"
            >
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message…"
                aria-label="Message"
                className="bg-input/40"
              />
              <Button type="submit" disabled={!input.trim() || typing}>
                Send
              </Button>
            </form>
          </div>
        </section>

        {/* Inspector */}
        <aside className="space-y-4">
          <div className="glass rounded-2xl p-4">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">NLP inspector</h2>
            <dl className="mt-3 space-y-2 font-mono text-xs">
              <Row label="intent" value={lastMeta?.intent ?? "—"} accent />
              <Row label="confidence" value={lastMeta ? lastMeta.confidence.toFixed(2) : "—"} />
              <Row label="sentiment" value={lastMeta?.sentiment ?? "—"} />
              <Row
                label="entities"
                value={lastMeta && Object.keys(lastMeta.entities).length ? Object.keys(lastMeta.entities).join(", ") : "none"}
              />
            </dl>
          </div>

          <div className="glass rounded-2xl p-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Slots</h2>
              <span className="font-mono text-xs text-accent">{filled}/{SLOT_ORDER.length}</span>
            </div>
            <div className="mt-3 h-1 overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full bg-accent transition-all duration-500"
                style={{ width: `${(filled / SLOT_ORDER.length) * 100}%` }}
              />
            </div>
            <ul className="mt-3 space-y-1.5 font-mono text-xs">
              {SLOT_ORDER.map((s) => (
                <li key={s} className="flex items-center justify-between gap-2">
                  <span className="text-muted-foreground">{s}</span>
                  <span className={state.data[s] ? "truncate text-foreground" : "text-muted-foreground/50"}>
                    {state.data[s] ?? "pending"}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <div className="glass rounded-2xl p-4">
            <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">registration.json</h2>
            <pre className="mt-3 max-h-56 overflow-auto rounded-lg bg-secondary/60 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
{JSON.stringify(state.completed ?? { status: "in_progress", ...state.data }, null, 2)}
            </pre>
          </div>
        </aside>
      </div>
    </main>
  );
}

function Row({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className={accent ? "truncate text-accent" : "truncate text-foreground"}>{value}</dd>
    </div>
  );
}

/** Minimal markdown: **bold** and line breaks. */
function Rich({ text }: { text: string }) {
  return (
    <>
      {text.split("\n").map((line, i) => (
        <p key={i} className={line ? "" : "h-2"}>
          {line.split(/(\*\*[^*]+\*\*)/g).map((part, j) =>
            part.startsWith("**") && part.endsWith("**") ? (
              <strong key={j} className="font-semibold">
                {part.slice(2, -2)}
              </strong>
            ) : (
              <span key={j}>{part}</span>
            ),
          )}
        </p>
      ))}
    </>
  );
}
