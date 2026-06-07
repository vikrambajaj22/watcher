import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { streamChat, type ChatEvent, type ChatMessage } from "../api/watcher";
import { AiBlurb } from "../components/AiBlurb";
import { MediaCard } from "../components/MediaCard";
import { VerdictBadge } from "../components/VerdictBadge";
import { posterUrl } from "../lib/poster";

type ToolStatus = { tool: string; label: string; done: boolean; data?: Record<string, unknown> };

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
  reasoning?: string;
  character?: string;
  watched_at?: string;
};

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
          <VerdictBadge willLike={Boolean(data.will_like)} score={Number(data.score ?? 0.5)} />
          {data.explanation ? <AiBlurb>{String(data.explanation)}</AiBlurb> : null}
          {item?.id && item.media_type && (
            <Link
              to={`/similar?id=${item.id}&type=${encodeURIComponent(item.media_type)}`}
              className="self-start text-xs text-accent hover:text-accent/80 transition-colors"
            >
              Find Similar →
            </Link>
          )}
        </div>
      </div>
    );
  }

  if (type === "actor_history") {
    const person = data.person as { name?: string; profile_path?: string } | undefined;
    return (
      <div className="mt-2">
        {person?.name && (
          <p className="text-xs text-muted mb-2">
            {person.name} — {items.length} title{items.length !== 1 ? "s" : ""} in your history
          </p>
        )}
        {items.length > 0 && (
          <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
            {items.slice(0, 8).map((item) => (
              <MediaCard
                key={`${item.id}-${item.media_type}`}
                id={item.id}
                title={item.title ?? "Unknown"}
                mediaType={item.media_type}
                posterPath={item.poster_path}
                subtitle={item.character ?? undefined}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  if (type === "history") {
    return (
      <div className="mt-2 grid grid-cols-3 sm:grid-cols-4 gap-2">
        {items.slice(0, 8).map((item) => (
          <MediaCard
            key={`${item.id}-${item.media_type}`}
            id={item.id}
            title={item.title ?? "Unknown"}
            mediaType={item.media_type}
            posterPath={item.poster_path}
          />
        ))}
      </div>
    );
  }

  if (items.length > 0) {
    return (
      <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2">
        {items.slice(0, 6).map((item) => (
          <MediaCard
            key={`${item.id}-${item.media_type}`}
            id={item.id}
            title={item.title ?? "Unknown"}
            mediaType={item.media_type}
            posterPath={item.poster_path}
            subtitle={
              type === "recommendations" ? (item.reasoning ?? undefined) : undefined
            }
            overview={type !== "recommendations" ? (item.overview ?? undefined) : undefined}
            similarLink
          />
        ))}
      </div>
    );
  }

  return null;
}

export function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  function clearChat() {
    setTurns([]);
    setInput("");
    setBusy(false);
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
        tools: [...t.tools, { tool: ev.tool, label: ev.label, done: false }],
      }));
    } else if (ev.type === "tool_result") {
      updateLast((t) => ({
        ...t,
        tools: t.tools.map((s) =>
          s.tool === ev.tool && !s.done
            ? { ...s, done: true, data: ev.data }
            : s,
        ),
      }));
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");

    const historyMessages: ChatMessage[] = turns
      .filter((t) => t.content)
      .map((t) => ({ role: t.role, content: t.content }));

    const newTurns: Turn[] = [
      ...turns,
      { id: crypto.randomUUID(), role: "user", content: text, tools: [] },
      { id: crypto.randomUUID(), role: "assistant", content: "", tools: [] },
    ];
    setTurns(newTurns);
    setBusy(true);

    try {
      for await (const ev of streamChat([...historyMessages, { role: "user", content: text }])) {
        if (ev.type === "done") break;
        handleEvent(ev);
      }
    } catch {
      updateLast((t) => ({ ...t, content: t.content || "Something went wrong. Please try again." }));
    } finally {
      setBusy(false);
      updateLast((t) => ({ ...t, content: t.content || "No response received." }));
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }

  return (
    <div className="w-full flex flex-col" style={{ minHeight: "calc(100vh - 12rem)" }}>
      <div className="flex items-center justify-between mb-1.5">
        <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">
          Chat with Watcher
        </h1>
        {turns.length > 0 && (
          <button
            type="button"
            onClick={clearChat}
            className="text-xs text-muted hover:text-text transition-colors bg-transparent border-0 cursor-pointer font-sans"
          >
            New chat
          </button>
        )}
      </div>
      <p className="text-muted mb-6">
        Ask for recommendations, check if you'll like something, or explore your history in natural language.
      </p>

      <div className="flex-1 flex flex-col gap-4 mb-4">
        {turns.length === 0 && (
          <div className="p-5 glass rounded-2xl text-muted text-sm flex flex-col gap-2">
            <p className="m-0 font-medium text-text">Try asking:</p>
            {[
              "Recommend me something to watch tonight",
              "Will I like Succession?",
              "Find me 90s sci-fi films I haven't seen",
              "What have I watched with Cate Blanchett?",
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
                    <div key={i} className="glass rounded-xl px-3 py-2">
                      <div className="flex items-center gap-2">
                        {!ts.done ? (
                          <div className="size-3.5 rounded-full border-2 border-border border-t-accent animate-spin [animation-duration:0.7s] shrink-0" aria-hidden />
                        ) : (
                          <div className="size-3.5 rounded-full bg-accent/20 border border-accent/40 shrink-0" />
                        )}
                        <span className="text-xs text-muted">{ts.label}</span>
                      </div>
                      {ts.done && ts.data && (
                        <ToolResultCards data={ts.data} />
                      )}
                    </div>
                  ))}
                  {turn.content && (
                    <div className="glass rounded-2xl rounded-tl-sm px-4 py-3">
                      <p className="text-sm m-0 whitespace-pre-wrap">{turn.content}</p>
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

      <div className="sticky bottom-0 pb-2 pt-2 bg-bg/80 backdrop-blur-sm">
        <div className="flex gap-2 items-end glass-dark rounded-2xl p-2">
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
