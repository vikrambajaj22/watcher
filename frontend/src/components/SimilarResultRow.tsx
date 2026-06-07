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
  const src = posterUrl(item.poster_path ?? null, "w185") ?? placeholderPoster(title);
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
    <article className="flex gap-4 sm:gap-5 p-4 sm:p-5 bg-surface border border-border rounded-xl transition-all duration-150 hover:border-accent/20 hover:shadow-lg hover:shadow-black/25">
      <div className="w-16 sm:w-[76px] shrink-0 rounded-lg overflow-hidden bg-bg shadow-md shadow-black/30">
        <img src={src} alt="" loading="lazy" className="w-full aspect-[2/3] object-cover block" />
      </div>

      <div className="flex-1 min-w-0">
        <span
          className={`inline-block text-[0.65rem] font-bold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded mb-1.5 ${
            isTv ? "bg-blue-400/10 text-blue-300" : "bg-emerald-400/10 text-emerald-300"
          }`}
        >
          {isTv ? "TV" : "Film"}
        </span>
        <h3 className="text-base font-semibold leading-snug tracking-[-0.02em] mb-1 line-clamp-2">
          {title}
        </h3>
        {overview && (
          <p className="text-sm text-muted leading-relaxed line-clamp-3 m-0">
            {String(item.overview).slice(0, 380)}
            {String(item.overview).length > 380 ? "…" : ""}
          </p>
        )}
        {will.status === "ok" && (
          <div className="mt-3 p-3 bg-accent/8 border border-accent/20 rounded-lg text-sm">
            {will.data.already_watched ? (
              <span className="font-semibold">Already watched</span>
            ) : (
              <>
                <span className="font-semibold block mb-1">
                  {(will.data.will_like ? "Likely Yes" : "Probably Not") +
                    (typeof will.data.score === "number"
                      ? ` (${Number(will.data.score).toFixed(2)})`
                      : "")}
                </span>
                <p className="text-muted m-0">{String(will.data.explanation ?? "")}</p>
              </>
            )}
          </div>
        )}
        {will.status === "err" && (
          <p className="text-sm text-danger mt-2 m-0">{will.message}</p>
        )}
      </div>

      <div className="hidden sm:flex flex-col gap-1.5 text-xs shrink-0 w-28">
        <div className="flex justify-between items-baseline gap-2">
          <span className="text-[0.72rem] uppercase tracking-[0.04em] text-muted">Rank</span>
          <span className="font-semibold">{rank}</span>
        </div>
        {item.release_date && (
          <div className="flex justify-between items-baseline gap-2">
            <span className="text-[0.72rem] uppercase tracking-[0.04em] text-muted">Year</span>
            <span className="font-semibold">{item.release_date.slice(0, 4)}</span>
          </div>
        )}
      </div>

      <div className="shrink-0 self-center flex flex-col gap-2">
        {tmdbUrl && (
          <a
            className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-sm font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 hover:no-underline transition-all whitespace-nowrap"
            href={tmdbUrl}
            target="_blank"
            rel="noreferrer"
          >
            Open in TMDB
          </a>
        )}
        {item.id && item.media_type && (
          <button
            type="button"
            className="inline-flex items-center justify-center px-3 py-1.5 rounded-lg text-xs font-semibold bg-transparent text-muted border border-border hover:text-text hover:border-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer whitespace-nowrap"
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
