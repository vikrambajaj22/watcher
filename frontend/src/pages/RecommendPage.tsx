import { LoadingBox } from "../components/LoadingBox";
import { ErrorBox } from "../components/ErrorBox";
import { useRef, useState } from "react";
import { apiFetch } from "../api/client";
import { type Recommendation } from "../api/watcher";
import { MediaCard } from "../components/MediaCard";
import { useWatchlist } from "../contexts/WatchlistContext";

type Media = "movie" | "tv" | "all";

const REC_CACHE_TTL_MS = 60_000;

const inputCls = "glass-input rounded-lg text-text px-2.5 py-2 text-[16px] sm:text-sm";

const GENRES = [
  "Action", "Adventure", "Animation", "Comedy", "Crime",
  "Documentary", "Drama", "Family", "Fantasy", "Horror",
  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
];

export function RecommendPage() {
  const { isOnWatchlist, toggle, isToggling } = useWatchlist();
  const [hideWatchlisted, setHideWatchlisted] = useState(false);
  const [media, setMedia] = useState<Media>("all");
  const [count, setCount] = useState(8);
  const [genres, setGenres] = useState<string[]>([]);
  const [items, setItems] = useState<Recommendation[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [cacheHit, setCacheHit] = useState(false);
  const cacheRef = useRef<{ key: string; at: number; items: Recommendation[] } | null>(null);

  function toggleGenre(g: string) {
    setGenres((prev) => prev.includes(g) ? prev.filter((x) => x !== g) : [...prev, g]);
  }

  async function run() {
    setErr(null);
    setCacheHit(false);
    const key = `${media}:${count}:${[...genres].sort().join(",")}`;
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
      const body: Record<string, unknown> = { recommend_count: count };
      if (genres.length > 0) body.genre_hint = genres;
      const r = await apiFetch(`/recommend/tmdb/${media}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
      <h1 className="page-title">Recommendations</h1>
      <p className="text-muted mb-6">
        Personalized picks from your history. The same settings within about a minute reuse the
        previous result.
      </p>

      <div className="p-4 glass-dark rounded-2xl mb-4">
        <div className="flex flex-wrap gap-3 items-end mb-4">
          <label className="flex flex-col gap-1">
            <span className="field-label">Media</span>
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
            <span className="field-label">Count</span>
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

        <div className="mb-4">
          <span className="field-label block mb-2">Genres <span className="text-muted font-normal">(optional)</span></span>
          <div className="flex flex-wrap gap-1.5">
            {GENRES.map((g) => {
              const active = genres.includes(g);
              return (
                <button
                  key={g}
                  type="button"
                  onClick={() => toggleGenre(g)}
                  className={`px-2.5 py-1 rounded-full text-xs font-medium transition-all cursor-pointer border font-sans ${
                    active
                      ? "bg-accent/20 text-accent border-accent/40"
                      : "bg-transparent text-muted border-border hover:text-text hover:border-muted"
                  }`}
                >
                  {g}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex flex-wrap gap-3 items-center">
          <button
            type="button"
            className="btn-primary"
            disabled={busy}
            onClick={() => void run()}
          >
            {busy ? "Generating…" : "Get Recommendations"}
          </button>
          <label className="flex items-center gap-2 text-sm text-muted cursor-pointer select-none">
            <input
              type="checkbox"
              checked={hideWatchlisted}
              onChange={(e) => setHideWatchlisted(e.target.checked)}
            />
            Hide watchlisted
          </label>
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

      {err && <ErrorBox message={err} />}
      {busy && <LoadingBox label="Generating Recommendations…" />}

      {!busy && items && items.length === 0 && (
        <p className="empty-state">
          No recommendations returned. Try a different mix or check that your watch history is synced.
        </p>
      )}

      {!busy && items && items.length > 0 && (() => {
        const visibleItems = hideWatchlisted
          ? items.filter((rec) => {
              const raw = rec.metadata?.id != null ? String(rec.metadata.id) : String(rec.id ?? "").trim();
              const id = Number(raw);
              return !(Number.isFinite(id) && id > 0 && rec.media_type && isOnWatchlist(id, rec.media_type));
            })
          : items;
        return (
        <>
          <div className="flex items-baseline gap-2 mb-3">
            <h2 className="text-lg font-semibold tracking-tight m-0">Your Picks</h2>
            <span className="text-sm text-muted">
              {visibleItems.length} title{visibleItems.length === 1 ? "" : "s"}
              {genres.length > 0 && ` · ${genres.join(", ")}`}
            </span>
          </div>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-4 sm:gap-5">
            {visibleItems.map((rec, index) => {
              const md = rec.metadata ?? {};
              const poster = (md.poster_path as string | undefined) ?? undefined;
              const overview = (md.overview as string | undefined) ?? null;
              const raw = md.id != null ? String(md.id) : String(rec.id ?? "").trim();
              const id = Number(raw);
              const idOk = Number.isFinite(id) && id > 0;
              return (
                <div key={`${rec.id}-${rec.title}-${index}`}>
                  <MediaCard
                    id={idOk ? id : 0}
                    title={rec.title}
                    mediaType={rec.media_type}
                    posterPath={poster}
                    subtitle={rec.reasoning}
                    overview={overview}
                    similarLink={idOk}
                    watchlistOn={idOk && rec.media_type ? isOnWatchlist(id, rec.media_type) : undefined}
                    watchlistLoading={idOk && rec.media_type ? isToggling(id, rec.media_type) : undefined}
                    onWatchlistToggle={
                      idOk && rec.media_type
                        ? () => void toggle({ id, title: rec.title, mediaType: rec.media_type!, posterPath: poster, overview })
                        : undefined
                    }
                  />
                </div>
              );
            })}
          </div>
        </>
        );
      })()}
    </div>
  );
}
