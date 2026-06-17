import { useEffect, useRef, useState, type ReactNode } from "react";

function uuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function CollapsibleDebug({ summary, children }: { summary: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs text-muted hover:text-text transition-colors cursor-pointer bg-transparent border-0 font-sans p-0"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={`transition-transform ${open ? "rotate-90" : ""}`}>
          <path d="m9 18 6-6-6-6"/>
        </svg>
        {summary}
      </button>
      {open && <div className="mt-1.5 text-xs text-muted leading-relaxed">{children}</div>}
    </div>
  );
}
import ReactMarkdown from "react-markdown";
import { streamChat, type ChatEvent } from "../api/watcher";
import { AiBlurb } from "../components/AiBlurb";
import { MediaCard } from "../components/MediaCard";
import { VerdictBadge } from "../components/VerdictBadge";
import { posterUrl } from "../lib/poster";

type ToolStatus = { tool: string; label: string; done: boolean; run_id?: string; args?: Record<string, unknown>; data?: Record<string, unknown>; duration_ms?: number };

type Turn = {
  id: string;
  role: "user" | "assistant";
  content: string;
  tools: ToolStatus[];
};

type MediaItem = {
  id: number;
  title?: string;
  media_type?: string;
  poster_path?: string | null;
  overview?: string | null;
  character?: string;
  watched_at?: string;
  watched?: boolean;
};

function ToolCallRow({ ts }: { ts: ToolStatus }) {
  const [open, setOpen] = useState(false);
  const hasDebug = ts.args && Object.keys(ts.args).length > 0;
  return (
    <div className="glass rounded-xl overflow-hidden">
      <button
        type="button"
        onClick={() => hasDebug && setOpen(o => !o)}
        className={`w-full flex items-center gap-2 px-3 py-2 text-left bg-transparent border-0 font-sans ${hasDebug ? "cursor-pointer hover:bg-white/[0.03] transition-colors" : "cursor-default"}`}
      >
        {!ts.done ? (
          <div className="size-3.5 rounded-full border-2 border-border border-t-accent animate-spin [animation-duration:0.7s] shrink-0" aria-hidden />
        ) : (
          <div className="size-3.5 rounded-full bg-accent/20 border border-accent/40 shrink-0" />
        )}
        <span className="text-xs text-muted flex-1">{ts.label}</span>
        {hasDebug && (
          <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={`text-muted transition-transform ${open ? "rotate-90" : ""}`}>
            <path d="m9 18 6-6-6-6"/>
          </svg>
        )}
      </button>
      {open && hasDebug && (
        <div className="border-t border-white/[0.06] px-3 py-2 flex flex-col gap-2">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wide text-muted/60 mb-0.5">args</p>
            <pre className="text-[10px] leading-relaxed whitespace-pre-wrap break-all font-mono text-muted">{JSON.stringify(ts.args, null, 2)}</pre>
          </div>
          {ts.done && ts.data && (
            <div>
              <div className="flex items-center justify-between mb-0.5">
                <p className="text-[10px] font-semibold uppercase tracking-wide text-muted/60">result</p>
                {ts.duration_ms !== undefined && (
                  <span className="text-[10px] text-muted/50">{ts.duration_ms}ms</span>
                )}
              </div>
              <pre className="text-[10px] leading-relaxed whitespace-pre-wrap break-all font-mono text-muted">{JSON.stringify(ts.data, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
      {ts.done && ts.data && (
        <div className="px-3 pb-2">
          <ToolResultCards data={ts.data} />
        </div>
      )}
    </div>
  );
}

function ToolResultCards({ data }: { data: Record<string, unknown> }) {
  const type = data.type as string;
  const items = (data.items ?? []) as MediaItem[];

  if (type === "error") {
    return (
      <p className="text-sm text-danger mt-2">{String(data.message ?? "Error")}</p>
    );
  }

  if (type === "will_like") {
    const item = data.item as { id?: number; title?: string; media_type?: string; poster_path?: string; overview?: string } | undefined;
    return (
      <div className="mt-2 flex flex-wrap gap-3 items-start">
        {item?.poster_path && (
          <img
            src={posterUrl(item.poster_path, "w185") ?? ""}
            alt=""
            className="w-12 rounded-md shrink-0"
            loading="lazy"
          />
        )}
        <div className="flex-1 min-w-0 flex flex-col gap-1.5">
          {data.already_watched ? (
            <span className="text-sm font-semibold text-emerald-300">Already watched</span>
          ) : (
            <>
              <VerdictBadge willLike={Boolean(data.will_like)} score={Number(data.score ?? 0.5)} />
              {data.explanation ? <AiBlurb>{String(data.explanation)}</AiBlurb> : null}
            </>
          )}
        </div>
      </div>
    );
  }

  if (type === "person_lookup") {
    const results = (data.results ?? []) as Array<{
      id: number;
      name?: string;
      profile_path?: string | null;
      known_for_department?: string | null;
      known_for?: string[];
    }>;
    return (
      <div className="mt-2 flex flex-col gap-2">
        {results.map((p) => (
          <div key={p.id} className="flex items-center gap-3">
            {p.profile_path ? (
              <img
                src={posterUrl(p.profile_path, "w185") ?? ""}
                alt=""
                className="size-9 rounded-full object-cover shrink-0"
                loading="lazy"
              />
            ) : (
              <div className="size-9 rounded-full bg-surface border border-border shrink-0 flex items-center justify-center text-sm text-muted">
                {p.name?.[0] ?? "?"}
              </div>
            )}
            <div className="min-w-0">
              <div className="text-sm font-semibold truncate">{p.name}</div>
              <div className="text-xs text-muted truncate">
                {p.known_for_department}
                {p.known_for_department && p.known_for?.length ? " · " : ""}
                {p.known_for?.join(", ")}
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (type === "actor_history") {
    const person = data.person as { name?: string } | undefined;
    return (
      <CollapsibleDebug summary={`${person?.name ?? "Actor"} · ${items.length} title${items.length !== 1 ? "s" : ""} in history`}>
        {items.map(i => `${i.title ?? "Unknown"}${i.character ? ` (${i.character})` : ""}`).join(", ") || "—"}
      </CollapsibleDebug>
    );
  }

  if (type === "history") {
    return (
      <CollapsibleDebug summary={`${items.length} history item${items.length !== 1 ? "s" : ""}`}>
        {items.map(i => i.title ?? "Unknown").join(", ") || "—"}
      </CollapsibleDebug>
    );
  }

  if (type === "cast") {
    const cast = (data.cast ?? []) as Array<{ name?: string; character?: string }>;
    return (
      <CollapsibleDebug summary={`Cast of ${String(data.title ?? "title")} · ${cast.length} listed`}>
        {cast.map(c => `${c.name ?? "Unknown"}${c.character ? ` (${c.character})` : ""}`).join(", ") || "—"}
      </CollapsibleDebug>
    );
  }

  if (type === "watchlist") {
    const titles = (data.titles ?? []) as Array<{ title?: string; media_type?: string }>;
    return (
      <CollapsibleDebug summary={`${titles.length} title${titles.length !== 1 ? "s" : ""} on your watchlist`}>
        {titles.map(t => t.title ?? "Unknown").join(", ") || "—"}
      </CollapsibleDebug>
    );
  }

  if (type === "title_details") {
    const genres = (data.genres ?? []) as string[];
    const runtime = data.runtime_minutes as number | null;
    const seasons = data.number_of_seasons as number | null;
    const meta = [
      data.year ? String(data.year) : null,
      genres.length ? genres.join(", ") : null,
      runtime ? `${runtime} min` : null,
      seasons ? `${seasons} season${seasons !== 1 ? "s" : ""}` : null,
      data.vote_average ? `★ ${data.vote_average}` : null,
    ].filter(Boolean);
    return (
      <div className="mt-2 flex flex-wrap gap-3 items-start">
        {data.poster_path ? (
          <img
            src={posterUrl(String(data.poster_path), "w185") ?? ""}
            alt=""
            className="w-14 rounded-md shrink-0"
            loading="lazy"
          />
        ) : null}
        <div className="flex-1 min-w-0 flex flex-col gap-1">
          <div className="text-sm font-semibold">{String(data.title ?? "Unknown")}</div>
          <div className="text-xs text-muted">{meta.join(" · ")}</div>
          {data.tagline ? <div className="text-xs italic text-muted/80">{String(data.tagline)}</div> : null}
          {data.overview ? <p className="text-xs text-muted leading-relaxed mt-0.5">{String(data.overview)}</p> : null}
        </div>
      </div>
    );
  }

  if (type === "taste_profile") {
    const genres = (data.genres ?? []) as string[];
    const themes = (data.themes ?? []) as string[];
    const avoid = (data.avoid ?? []) as string[];
    const Chips = ({ label, values }: { label: string; values: string[] }) =>
      values.length ? (
        <div className="flex flex-wrap gap-1 items-center">
          <span className="text-[10px] uppercase tracking-wide text-muted/60">{label}</span>
          {values.map(v => (
            <span key={v} className="text-[11px] px-1.5 py-0.5 rounded-md bg-white/[0.06] text-muted">{v}</span>
          ))}
        </div>
      ) : null;
    return (
      <div className="mt-2 flex flex-col gap-1.5">
        {data.signature ? <div className="text-sm font-semibold">{String(data.signature)}</div> : null}
        {data.summary ? <p className="text-xs text-muted leading-relaxed">{String(data.summary)}</p> : null}
        <Chips label="genres" values={genres} />
        <Chips label="themes" values={themes} />
        <Chips label="avoids" values={avoid} />
      </div>
    );
  }

  if (items.length > 0) {
    const watchedExcluded = Number(data.watched_excluded ?? 0);
    return (
      <div className="mt-2">
        {type === "discover" && watchedExcluded > 0 && (
          <p className="text-xs text-muted mb-2">{watchedExcluded} already-watched title{watchedExcluded !== 1 ? "s" : ""} excluded</p>
        )}
        <div className="grid grid-cols-3 sm:grid-cols-4 gap-1.5">
          {items.map((item) => (
            <MediaCard
              key={`${item.id}-${item.media_type}`}
              id={item.id}
              title={item.title ?? "Unknown"}
              mediaType={item.media_type}
              posterPath={item.poster_path}
              overview={type !== "recommendations" ? (item.overview ?? undefined) : undefined}
              watched={item.watched}
              compact
            />
          ))}
        </div>
      </div>
    );
  }

  return null;
}

export function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [threadId, setThreadId] = useState(() => uuid());
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  function clearChat() {
    setTurns([]);
    setInput("");
    setBusy(false);
    setThreadId(uuid());
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  function updateLast(updater: (prev: Turn) => Turn) {
    setTurns((prev) => {
      if (!prev.length) return prev;
      const last = prev[prev.length - 1];
      return [...prev.slice(0, -1), updater(last)];
    });
  }

  function handleEvent(ev: ChatEvent) {
    if (ev.type === "message") {
      updateLast((t) => ({ ...t, content: t.content + ev.content }));
    } else if (ev.type === "tool_start") {
      updateLast((t) => ({
        ...t,
        tools: [...t.tools, { tool: ev.tool, label: ev.label, done: false, run_id: ev.run_id, args: ev.args }],
      }));
    } else if (ev.type === "tool_result") {
      updateLast((t) => ({
        ...t,
        tools: t.tools.map((s) =>
          (ev.run_id ? s.run_id === ev.run_id : s.tool === ev.tool && !s.done)
            ? { ...s, done: true, data: ev.data, duration_ms: ev.duration_ms }
            : s,
        ),
      }));
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");

    const newTurns: Turn[] = [
      ...turns,
      { id: uuid(), role: "user", content: text, tools: [] },
      { id: uuid(), role: "assistant", content: "", tools: [] },
    ];
    setTurns(newTurns);
    setBusy(true);

    try {
      for await (const ev of streamChat(threadId, text)) {
        if (ev.type === "done") break;
        handleEvent(ev);
      }
    } catch {
      updateLast((t) => ({ ...t, content: t.content || "Something went wrong. Please try again." }));
    } finally {
      setBusy(false);
      updateLast((t) => ({ ...t, content: t.content || (t.tools.some(ts => ts.done) ? "" : "No response received.") }));
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }

  return (
    <div className="flex flex-col h-full pt-7">
      <div className="mb-1.5 shrink-0">
        <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">
          Chat with Watcher
        </h1>
      </div>
      <p className="text-muted mb-6 shrink-0">
        Ask for recommendations, check if you'll like something, or explore your history in natural language.
      </p>

      <div className="flex-1 overflow-y-auto min-h-0 flex flex-col gap-4 pb-2">
        {turns.length === 0 && (
          <div className="p-5 glass rounded-2xl text-muted text-sm flex flex-col gap-2">
            <p className="m-0 font-medium text-text">Try asking:</p>
            {[
              "Recommend me something to watch tonight",
              "Will I like <title>?",
              "Find me 90s sci-fi films I haven't seen",
              "What have I watched with <actor>?",
            ].map((s) => (
              <button
                key={s}
                type="button"
                className="text-left text-sm text-muted hover:text-text transition-colors cursor-pointer bg-transparent border-0 font-sans p-0"
                onClick={() => { setInput(s); inputRef.current?.focus(); }}
              >
                "{s}"
              </button>
            ))}
          </div>
        )}

        {turns.map((turn) => (
          <div
            key={turn.id}
            className={`flex ${turn.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] ${
                turn.role === "user"
                  ? "glass-dark rounded-2xl rounded-tr-sm px-4 py-3"
                  : "flex flex-col gap-2 w-full"
              }`}
            >
              {turn.role === "user" ? (
                <p className="text-sm m-0">{turn.content}</p>
              ) : (
                <>
                  {turn.tools.map((ts, i) => (
                    <ToolCallRow key={i} ts={ts} />
                  ))}
                  {turn.content && (
                    <div className="glass rounded-2xl rounded-tl-sm px-4 py-3 text-sm prose prose-invert prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-2 prose-code:text-accent prose-pre:bg-white/5 prose-pre:rounded-lg prose-a:text-accent">
                      <ReactMarkdown>{turn.content}</ReactMarkdown>
                    </div>
                  )}
                  {!turn.content && !turn.tools.length && busy && (
                    <div className="glass rounded-2xl rounded-tl-sm px-4 py-3">
                      <div className="size-4 rounded-full border-2 border-border border-t-accent animate-spin [animation-duration:0.7s]" aria-hidden />
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="shrink-0 pb-4 pt-2 bg-bg/80 backdrop-blur-sm">
        <div className="flex gap-2 items-end glass-dark rounded-2xl p-2">
          <button
            type="button"
            onClick={clearChat}
            disabled={turns.length === 0}
            className="inline-flex items-center justify-center size-9 rounded-xl bg-white/[0.06] hover:bg-white/[0.10] text-muted hover:text-text transition-all disabled:opacity-30 disabled:cursor-not-allowed border-0 shrink-0 cursor-pointer"
            title="New chat"
            aria-label="New chat"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
            </svg>
          </button>
          <textarea
            ref={inputRef}
            className="flex-1 bg-transparent text-[16px] sm:text-sm text-text resize-none outline-none px-2 py-1.5 max-h-32 min-h-[2.5rem]"
            placeholder="Ask about movies and TV shows…"
            rows={1}
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <button
            type="button"
            className="inline-flex items-center justify-center size-9 rounded-xl bg-gradient-to-br from-accent to-accent-dim text-bg font-semibold cursor-pointer transition-all hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed border-0 shrink-0"
            disabled={busy || !input.trim()}
            onClick={() => void send()}
            aria-label="Send"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5"/><path d="m5 12 7-7 7 7"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
