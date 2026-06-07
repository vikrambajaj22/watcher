import { useState } from "react";
import { Link } from "react-router-dom";
import { apiFetch } from "../api/client";
import { type SearchHit, type WillLikeResponse } from "../api/watcher";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { posterUrl, placeholderPoster } from "../lib/poster";

export function WillLikePage() {
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [result, setResult] = useState<WillLikeResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!selectedHit) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await apiFetch("/will-like", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tmdb_id: selectedHit.id, media_type: selectedHit.media_type }),
      });
      const text = await r.text();
      let j: unknown;
      try { j = JSON.parse(text); } catch { j = { detail: text }; }
      if (!r.ok) {
        setErr(
          typeof j === "object" && j && "detail" in j
            ? String((j as { detail: unknown }).detail)
            : text,
        );
        return;
      }
      setResult(j as WillLikeResponse);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const displayTitle = result?.item.title ?? result?.item.name ?? "—";
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

      <div className="card form-card will-form-card">
        <label className="field field-block">
          <span className="field-label">Title</span>
          <SearchTypeahead
            selected={selectedHit}
            onSelect={(hit) => { setSelectedHit(hit); setResult(null); setErr(null); }}
            onClear={() => setSelectedHit(null)}
            placeholder="Search movie or show…"
          />
        </label>
        <div className="actions will-form-actions">
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || !selectedHit}
            onClick={() => void submit()}
          >
            {busy ? "Checking…" : "Check"}
          </button>
        </div>
      </div>

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
              src={posterUrl(result.item.poster_path, "w185") ?? placeholderPoster(displayTitle)}
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
                : `Score: ${(result.score * 100).toFixed(0)}%`}
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
            {itemId != null && itemId > 0 && result.item.media_type && (
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
