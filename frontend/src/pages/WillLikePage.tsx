import { useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch } from "../api/client";
import { type WillLikeResponse } from "../api/watcher";
import { posterUrl, placeholderPoster } from "../lib/poster";

type Mode = "title" | "id";

export function WillLikePage() {
  const [mode, setMode] = useState<Mode>("title");
  const [title, setTitle] = useState("");
  const [tmdbId, setTmdbId] = useState(550);
  const [mediaType, setMediaType] = useState<"movie" | "tv">("movie");
  const [result, setResult] = useState<WillLikeResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [disambig, setDisambig] = useState(false);

  async function submit(overrideMedia?: "movie" | "tv") {
    setBusy(true);
    setErr(null);
    setDisambig(false);
    setResult(null);
    const mt = overrideMedia ?? mediaType;
    const payload =
      mode === "title"
        ? { title: title.trim(), media_type: mt }
        : { tmdb_id: tmdbId, media_type: mt };

    try {
      const r = await apiFetch("/mcp/will-like", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const text = await r.text();
      let j: unknown;
      try {
        j = JSON.parse(text);
      } catch {
        j = { detail: text };
      }
      if (r.status === 400) {
        const detail = JSON.stringify(j);
        if (
          detail.includes("input_media_type") ||
          detail.toLowerCase().includes("ambiguous")
        ) {
          setDisambig(true);
          return;
        }
        setErr(
          typeof j === "object" && j && "detail" in j
            ? String((j as { detail: unknown }).detail)
            : text,
        );
        return;
      }
      if (!r.ok) {
        setErr(text);
        return;
      }
      setResult(j as WillLikeResponse);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const displayTitle =
    result?.item.title ?? result?.item.name ?? "—";
  const itemId = result?.item.id;
  const itemMt = (result?.item.media_type ?? "movie").toLowerCase();
  const isTv = itemMt === "tv";

  return (
    <div className="page page-wide">
      <h1 className="page-title">Will I Like It?</h1>
      <p className="lede">
        A short model-assisted read on whether a title fits your taste, based on
        your watch history.
      </p>

      <div className="tabs">
        <button
          type="button"
          className={mode === "title" ? "tab active" : "tab"}
          onClick={() => setMode("title")}
        >
          By Title
        </button>
        <button
          type="button"
          className={mode === "id" ? "tab active" : "tab"}
          onClick={() => setMode("id")}
        >
          By TMDB ID
        </button>
      </div>

      <div className="card form-card will-form-card">
        <div
          className={
            mode === "title"
              ? "will-form-grid"
              : "will-form-grid will-form-grid--id"
          }
        >
          {mode === "title" ? (
            <label className="field will-form-field-main">
              <span className="field-label">Title</span>
              <input
                className="input"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Movie Or Show Name"
              />
            </label>
          ) : (
            <label className="field will-form-field-main">
              <span className="field-label">TMDB ID</span>
              <input
                className="input input-narrow"
                type="number"
                min={1}
                value={tmdbId}
                onChange={(e) => setTmdbId(Number(e.target.value))}
              />
            </label>
          )}
          <label className="field will-form-field-type">
            <span className="field-label">Type</span>
            <select
              className="input"
              value={mediaType}
              onChange={(e) =>
                setMediaType(e.target.value as "movie" | "tv")
              }
            >
              <option value="movie">Movie</option>
              <option value="tv">TV</option>
            </select>
          </label>
        </div>
        <div className="actions will-form-actions">
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || (mode === "title" && !title.trim())}
            onClick={() => void submit()}
          >
            {busy ? "Checking…" : "Check"}
          </button>
        </div>
      </div>

      {disambig && (
        <div className="card card-warn">
          <p>This ID or title may match both movie and TV. Pick a type and retry.</p>
          <div className="actions">
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => void submit("movie")}
            >
              Retry As Movie
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => void submit("tv")}
            >
              Retry As TV
            </button>
          </div>
        </div>
      )}

      {err && <div className="card card-error">{err}</div>}

      {busy && (
        <div className="card loading-panel" role="status" aria-live="polite">
          <div className="loading-spinner" aria-hidden />
          <p className="loading-panel-text">Checking Your Taste Fit…</p>
        </div>
      )}

      {!busy && result && (
        <article className="history-card will-result-card">
          <div className="history-card-poster">
            <img
              src={
                posterUrl(result.item.poster_path, "w185") ??
                placeholderPoster(displayTitle)
              }
              alt=""
              loading="lazy"
            />
          </div>
          <div className="history-card-main">
            {result.already_watched ? (
              <span className="history-card-kind movie">Watched</span>
            ) : (
              <span className={`history-card-kind ${isTv ? "tv" : "movie"}`}>
                {isTv ? "TV" : "Film"}
              </span>
            )}
            <h2 className="history-card-title will-result-title">{displayTitle}</h2>
            <p className="history-card-sub muted">
              {result.already_watched
                ? "Already in your history."
                : `Score: ${result.score.toFixed(3)}`}
            </p>
            <p className="media-card-reasoning will-result-explanation">
              {result.explanation}
            </p>
          </div>
          <ul className="history-card-statline">
            <li>
              <span className="stat-label">Verdict</span>
              <span className="stat-value">
                {result.already_watched ? (
                  <span className="badge badge-ok">Seen</span>
                ) : result.will_like ? (
                  <span className="badge badge-ok">Likely Yes</span>
                ) : (
                  <span className="badge badge-warn">Probably Not</span>
                )}
              </span>
            </li>
          </ul>
          <div className="history-card-actions similar-result-actions">
            {itemId != null &&
              itemId > 0 &&
              result.item.media_type && (
                <Link
                  className="btn btn-secondary history-card-link"
                  to={`/similar?id=${itemId}&type=${encodeURIComponent(itemMt)}`}
                >
                  Find Similar
                </Link>
              )}
          </div>
        </article>
      )}
    </div>
  );
}
