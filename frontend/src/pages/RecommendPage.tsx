import { useRef, useState } from "react";
import { apiFetch } from "../api/client";
import { type Recommendation } from "../api/watcher";
import { MediaCard } from "../components/MediaCard";

type Media = "movie" | "tv" | "all";

const REC_CACHE_TTL_MS = 60_000;

export function RecommendPage() {
  const [media, setMedia] = useState<Media>("all");
  const [count, setCount] = useState(8);
  const [items, setItems] = useState<Recommendation[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [cacheHit, setCacheHit] = useState(false);
  const cacheRef = useRef<{
    key: string;
    at: number;
    items: Recommendation[];
  } | null>(null);

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
      const r = await apiFetch(`/recommend/${media}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recommend_count: count }),
      });
      if (!r.ok) {
        setErr(await r.text());
        return;
      }
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

  const mediaLabel =
    media === "all" ? "All" : media === "movie" ? "Movies" : "TV";

  return (
    <div className="page page-wide">
      <h1 className="page-title">Recommendations</h1>
      <p className="lede">
        Personalized picks from your history. The same settings within about a
        minute reuse the previous result for speed.
      </p>

      <div className="toolbar card">
        <div className="toolbar-row">
          <label className="field">
            <span className="field-label">Media</span>
            <select
              className="input"
              value={media}
              onChange={(e) => setMedia(e.target.value as Media)}
            >
              <option value="all">All</option>
              <option value="movie">Movies</option>
              <option value="tv">TV</option>
            </select>
          </label>
          <label className="field field-grow">
            <span className="field-label">How Many: {count}</span>
            <input
              className="range-input range-input-wide"
              type="range"
              min={1}
              max={20}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="toolbar-row">
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => void run()}
          >
            {busy ? "Generating…" : "Get Recommendations"}
          </button>
          <span className="muted toolbar-stats">
            {items && items.length > 0 && !busy
              ? `${items.length} title${items.length === 1 ? "" : "s"} · ${mediaLabel}`
              : `${mediaLabel} · Up To ${count} Results`}
          </span>
        </div>
      </div>

      {cacheHit && (
        <p className="hint rec-cache-hint">
          Showing the last result from under a minute ago. Run again for a fresh
          fetch.
        </p>
      )}

      {err && <div className="card card-error">{err}</div>}

      {busy && (
        <div className="card loading-panel" role="status" aria-live="polite">
          <div className="loading-spinner" aria-hidden />
          <p className="loading-panel-text">Generating Recommendations…</p>
        </div>
      )}

      {!busy && items && items.length === 0 && (
        <p className="muted card empty-results-note">
          No recommendations returned. Try a different mix or check that your
          watch history is synced.
        </p>
      )}

      {!busy && items && items.length > 0 && (
        <>
          <h2 className="section-title rec-results-heading">
            Your Picks{" "}
            <span className="rec-results-count muted">
              {items.length} title{items.length === 1 ? "" : "s"}
            </span>
          </h2>
          <div className="rec-media-grid">
            {items.map((rec, index) => {
              const md = rec.metadata ?? {};
              const poster =
                (md.poster_path as string | undefined) ?? undefined;
              const overview = (md.overview as string | undefined) ?? null;
              const raw =
                md.id != null ? String(md.id) : String(rec.id ?? "").trim();
              const id = Number(raw);
              const idOk = Number.isFinite(id) && id > 0;
              return (
                <div key={`${rec.id}-${rec.title}-${index}`} className="rec-card-wrap">
                  <span className="rec-card-rank" aria-hidden>
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
