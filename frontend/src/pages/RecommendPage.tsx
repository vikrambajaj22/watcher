import { useRef, useState } from "react";
import { apiFetch } from "../api/client";
import { type Recommendation } from "../api/watcher";
import { MediaCard } from "../components/MediaCard";

type Media = "movie" | "tv" | "all";

const REC_CACHE_TTL_MS = 60_000;

const inputCls =
  "bg-bg border border-border rounded-lg text-text px-2.5 py-2 font-sans text-sm outline-none transition-colors focus:border-accent/50";

export function RecommendPage() {
  const [media, setMedia] = useState<Media>("all");
  const [count, setCount] = useState(8);
  const [items, setItems] = useState<Recommendation[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [cacheHit, setCacheHit] = useState(false);
  const cacheRef = useRef<{ key: string; at: number; items: Recommendation[] } | null>(null);

  async function run() {
    setErr(null);
    setCacheHit(false);
    const key = `${media}:${count}`;
    const now = Date.now();
    const c = cacheRef.current;
    if (c && c.key === key && now - c.at < REC_CACHE_TTL_MS) {
      setItems(c.items);
      setCacheHit(true);
      return;
    }

    setBusy(true);
    setItems(null);
    try {
      const r = await apiFetch(`/recommend/tmdb/${media}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recommend_count: count }),
      });
      if (!r.ok) { setErr(await r.text()); return; }
      const j = (await r.json()) as { recommendations: Recommendation[] };
      const list = j.recommendations ?? [];
      setItems(list);
      cacheRef.current = { key, at: Date.now(), items: list };
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const mediaLabel = media === "all" ? "All" : media === "movie" ? "Movies" : "TV";

  return (
    <div className="w-full">
      <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-1.5 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">Recommendations</h1>
      <p className="text-muted mb-6">
        Personalized picks from your history. The same settings within about a minute reuse the
        previous result.
      </p>

      <div className="p-4 bg-surface border border-border rounded-xl mb-4">
        <div className="flex flex-wrap gap-3 items-end mb-3">
          <label className="flex flex-col gap-1">
            <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
              Media
            </span>
            <select
              className={inputCls}
              value={media}
              onChange={(e) => setMedia(e.target.value as Media)}
            >
              <option value="all">All</option>
              <option value="movie">Movies</option>
              <option value="tv">TV</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
              Count
            </span>
            <select
              className={inputCls}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
            >
              {[4, 6, 8, 12, 16, 20].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="flex flex-wrap gap-3 items-center">
          <button
            type="button"
            className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-white font-semibold text-sm cursor-pointer transition-all hover:brightness-110 hover:shadow-[0_0_20px_-4px] hover:shadow-accent/40 disabled:opacity-50 disabled:cursor-not-allowed border-0"
            disabled={busy}
            onClick={() => void run()}
          >
            {busy ? "Generating…" : "Get Recommendations"}
          </button>
          <span className="text-muted text-sm">
            {items && items.length > 0 && !busy
              ? `${items.length} title${items.length === 1 ? "" : "s"} · ${mediaLabel}`
              : `${mediaLabel} · Up to ${count}`}
          </span>
        </div>
      </div>

      {cacheHit && (
        <p className="text-sm text-muted mb-4">
          Showing cached result from under a minute ago. Run again for a fresh fetch.
        </p>
      )}

      {err && (
        <div className="p-4 bg-surface border border-danger/40 rounded-xl mb-4">
          <strong className="text-danger">Error: </strong>
          {err}
        </div>
      )}

      {busy && (
        <div className="flex items-center gap-4 p-5 bg-surface border border-border rounded-xl mb-4" role="status" aria-live="polite">
          <div className="size-7 rounded-full border-[3px] border-border border-t-accent animate-spin [animation-duration:0.7s] shrink-0" aria-hidden />
          <p className="text-sm m-0">Generating Recommendations…</p>
        </div>
      )}

      {!busy && items && items.length === 0 && (
        <p className="text-muted p-5 bg-surface border border-border rounded-xl">
          No recommendations returned. Try a different mix or check that your watch history is synced.
        </p>
      )}

      {!busy && items && items.length > 0 && (
        <>
          <div className="flex items-baseline gap-2 mb-3">
            <h2 className="text-lg font-semibold tracking-tight m-0">Your Picks</h2>
            <span className="text-sm text-muted">
              {items.length} title{items.length === 1 ? "" : "s"}
            </span>
          </div>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4 sm:gap-5">
            {items.map((rec, index) => {
              const md = rec.metadata ?? {};
              const poster = (md.poster_path as string | undefined) ?? undefined;
              const overview = (md.overview as string | undefined) ?? null;
              const raw = md.id != null ? String(md.id) : String(rec.id ?? "").trim();
              const id = Number(raw);
              const idOk = Number.isFinite(id) && id > 0;
              return (
                <div key={`${rec.id}-${rec.title}-${index}`} className="relative">
                  <span
                    className="absolute top-2 left-2 z-[1] min-w-[1.6rem] h-[1.6rem] px-1.5 flex items-center justify-center bg-bg/85 border border-border rounded-lg text-xs font-bold shadow-md shadow-black/30"
                    aria-hidden
                  >
                    {index + 1}
                  </span>
                  <MediaCard
                    id={idOk ? id : 0}
                    title={rec.title}
                    mediaType={rec.media_type}
                    posterPath={poster}
                    subtitle={rec.reasoning}
                    overview={overview}
                    similarLink={idOk}
                  />
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
