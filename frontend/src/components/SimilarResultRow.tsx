import { useState } from "react";
import { apiFetch } from "../api/client";
import type { SimilarResult } from "../api/watcher";
import { placeholderPoster, posterUrl } from "../lib/poster";

type WillState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: Record<string, unknown> }
  | { status: "err"; message: string };

export function SimilarResultRow({
  item,
  rank,
}: {
  item: SimilarResult;
  rank: number;
}) {
  const [will, setWill] = useState<WillState>({ status: "idle" });
  const title = item.title ?? String(item.id);
  const src =
    posterUrl(item.poster_path ?? null, "w185") ?? placeholderPoster(title);
  const mtRaw = (item.media_type || "movie").toLowerCase();
  const isTv = mtRaw === "tv";
  const tmdbUrl =
    item.id && item.media_type
      ? `https://www.themoviedb.org/${mtRaw}/${item.id}`
      : null;

  async function checkWillLike() {
    if (!item.id || !item.media_type) return;
    setWill({ status: "loading" });
    try {
      const r = await apiFetch("/will-like", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tmdb_id: item.id,
          media_type: item.media_type === "tv" ? "tv" : "movie",
        }),
      });
      const raw = await r.text();
      let j: Record<string, unknown>;
      try {
        j = JSON.parse(raw) as Record<string, unknown>;
      } catch {
        setWill({ status: "err", message: raw.slice(0, 200) });
        return;
      }
      if (!r.ok) {
        setWill({
          status: "err",
          message: String(j.detail ?? j.error ?? raw).slice(0, 300),
        });
        return;
      }
      setWill({ status: "ok", data: j });
    } catch (e) {
      setWill({
        status: "err",
        message: e instanceof Error ? e.message : "Request failed",
      });
    }
  }

  const overview = item.overview && String(item.overview).trim();

  return (
    <article className="history-card similar-result-card">
      <div className="history-card-poster">
        <img src={src} alt="" loading="lazy" />
      </div>
      <div className="history-card-main">
        <span className={`history-card-kind ${isTv ? "tv" : "movie"}`}>
          {isTv ? "TV" : "Film"}
        </span>
        <h3 className="history-card-title">{title}</h3>
        {overview ? (
          <p className="history-card-sub muted similar-result-overview">
            {String(item.overview).slice(0, 380)}
            {String(item.overview).length > 380 ? "…" : ""}
          </p>
        ) : null}
        {will.status === "ok" && (
          <div className="similar-will-snippet">
            {will.data.already_watched ? (
              <strong>Already watched</strong>
            ) : (
              <>
                <strong>
                  {(will.data.will_like ? "Likely Yes" : "Probably Not") +
                    (typeof will.data.score === "number"
                      ? ` (${Number(will.data.score).toFixed(2)})`
                      : "")}
                </strong>
                <p className="muted">{String(will.data.explanation ?? "")}</p>
              </>
            )}
          </div>
        )}
        {will.status === "err" && (
          <p className="hint similar-will-err">{will.message}</p>
        )}
      </div>
      <ul className="history-card-statline">
        <li>
          <span className="stat-label">Rank</span>
          <span className="stat-value">{rank}</span>
        </li>
        {item.release_date && (
          <li>
            <span className="stat-label">Year</span>
            <span className="stat-value">{item.release_date.slice(0, 4)}</span>
          </li>
        )}
      </ul>
      <div className="history-card-actions similar-result-actions">
        {tmdbUrl && (
          <a
            className="btn btn-secondary history-card-link"
            href={tmdbUrl}
            target="_blank"
            rel="noreferrer"
          >
            Open In TMDB
          </a>
        )}
        {item.id && item.media_type && (
          <button
            type="button"
            className="btn btn-ghost btn-small"
            disabled={will.status === "loading"}
            onClick={() => void checkWillLike()}
          >
            {will.status === "loading" ? "Checking…" : "Will I Like?"}
          </button>
        )}
      </div>
    </article>
  );
}
