import { useState } from "react";
import { apiFetch } from "../api/client";
import {
  type Recommendation,
  type RecommendResponse,
  type TmdbRecommendResponse,
} from "../api/watcher";
import { MediaCard } from "../components/MediaCard";

type Media = "movie" | "tv" | "all";

function RecColumn({
  title,
  subtitle,
  items,
  err,
  busy,
}: {
  title: string;
  subtitle: string;
  items: Recommendation[] | null;
  err: string | null;
  busy: boolean;
}) {
  return (
    <section className="rec-compare-col card">
      <h2 className="rec-compare-col-title">{title}</h2>
      <p className="muted rec-compare-col-lede">{subtitle}</p>
      {err && <div className="card card-error">{err}</div>}
      {busy && (
        <div className="loading-panel loading-panel-compact" role="status">
          <div className="loading-spinner" aria-hidden />
          <p className="loading-panel-text">Working…</p>
        </div>
      )}
      {!busy && items && items.length === 0 && (
        <p className="muted">No results.</p>
      )}
      {!busy && items && items.length > 0 && (
        <div className="rec-media-grid rec-compare-grid">
          {items.map((rec, index) => {
            const md = rec.metadata ?? {};
            const poster = (md.poster_path as string | undefined) ?? undefined;
            const overview = (md.overview as string | undefined) ?? null;
            const raw =
              md.id != null ? String(md.id) : String(rec.id ?? "").trim();
            const id = Number(raw);
            const idOk = Number.isFinite(id) && id > 0;
            return (
              <div
                key={`${title}-${rec.id}-${index}`}
                className="rec-card-wrap"
              >
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
      )}
    </section>
  );
}

export function RecommendComparePage() {
  const [media, setMedia] = useState<Media>("all");
  const [count, setCount] = useState(6);
  const [faissItems, setFaissItems] = useState<Recommendation[] | null>(null);
  const [tmdbItems, setTmdbItems] = useState<Recommendation[] | null>(null);
  const [tmdbDebug, setTmdbDebug] = useState<Record<string, unknown> | null>(
    null,
  );
  const [faissErr, setFaissErr] = useState<string | null>(null);
  const [tmdbErr, setTmdbErr] = useState<string | null>(null);
  const [busyFaiss, setBusyFaiss] = useState(false);
  const [busyTmdb, setBusyTmdb] = useState(false);

  async function runBoth() {
    setFaissErr(null);
    setTmdbErr(null);
    setTmdbDebug(null);
    setFaissItems(null);
    setTmdbItems(null);

    const body = JSON.stringify({ recommend_count: count });

    setBusyFaiss(true);
    setBusyTmdb(true);

    const faissP = apiFetch(`/recommend/${media}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        const j = (await r.json()) as RecommendResponse;
        setFaissItems(j.recommendations ?? []);
      })
      .catch((e) => {
        setFaissErr(e instanceof Error ? e.message : "FAISS path failed");
      })
      .finally(() => setBusyFaiss(false));

    const tmdbP = apiFetch(`/recommend/tmdb/${media}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text());
        const j = (await r.json()) as TmdbRecommendResponse;
        setTmdbItems(j.recommendations ?? []);
        setTmdbDebug(j.debug ?? null);
      })
      .catch((e) => {
        setTmdbErr(e instanceof Error ? e.message : "TMDB path failed");
      })
      .finally(() => setBusyTmdb(false));

    await Promise.all([faissP, tmdbP]);
  }

  return (
    <div className="page page-wide">
      <h1 className="page-title">Compare Recommendations</h1>
      <p className="lede">
        Side-by-side: classic FAISS + LLM picker vs experimental{" "}
        <strong>watch history → LLM plan → TMDB API → LLM picker</strong> (no
        metadata DB or index required).
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
            <span className="field-label">How many: {count}</span>
            <input
              className="range-input range-input-wide"
              type="range"
              min={1}
              max={12}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
            />
          </label>
        </div>
        <div className="toolbar-row">
          <button
            type="button"
            className="btn btn-primary"
            disabled={busyFaiss || busyTmdb}
            onClick={() => void runBoth()}
          >
            {busyFaiss || busyTmdb ? "Running both…" : "Compare both"}
          </button>
        </div>
      </div>

      {tmdbDebug && (
        <details className="card rec-compare-debug">
          <summary>TMDB path debug</summary>
          <pre className="json-pre admin-raw-pre">
            {JSON.stringify(tmdbDebug, null, 2)}
          </pre>
        </details>
      )}

      <div className="rec-compare-columns">
        <RecColumn
          title="FAISS (current)"
          subtitle="Vector index over local metadata, then LLM picks from candidates."
          items={faissItems}
          err={faissErr}
          busy={busyFaiss}
        />
        <RecColumn
          title="TMDB + LLM (beta)"
          subtitle="LLM plans discover queries; TMDB fills candidates; LLM picks."
          items={tmdbItems}
          err={tmdbErr}
          busy={busyTmdb}
        />
      </div>
    </div>
  );
}
