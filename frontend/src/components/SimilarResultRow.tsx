import { useState } from "react";
import { apiFetch } from "../api/client";
import type { SimilarResult } from "../api/watcher";
import { AiBlurb } from "./AiBlurb";
import { ExternalMediaLink } from "./ExternalMediaLink";
import { VerdictBadge } from "./VerdictBadge";
import { placeholderPoster, posterUrl } from "../lib/poster";
import { useWatchlist } from "../contexts/WatchlistContext";

type WillState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: Record<string, unknown> }
  | { status: "err"; message: string };

export function SimilarResultRow({ item, hideWillLike, hideWatchlist }: { item: SimilarResult; hideWillLike?: boolean; hideWatchlist?: boolean }) {
  const { isOnWatchlist, toggle, isToggling } = useWatchlist();
  const [will, setWill] = useState<WillState>({ status: "idle" });
  const title = item.title ?? String(item.id);
  const src = posterUrl(item.poster_path ?? null, "w185") ?? placeholderPoster();
  const mtRaw = (item.media_type || "movie").toLowerCase();
  const isTv = mtRaw === "tv";

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
          title: item.title ?? undefined,
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

  const willOutput = will.status === "ok" || will.status === "err";

  return (
    <article className="flex items-start gap-4 sm:gap-5 p-4 sm:p-5 glass glass-hover rounded-2xl flex-wrap">
      <div className="w-16 sm:w-[76px] shrink-0 rounded-lg overflow-hidden bg-bg shadow-md shadow-black/30 relative">
        <img src={src} alt="" loading="lazy" className="w-full aspect-[2/3] object-contain block" />
        {item.id > 0 && (
          <div className="absolute top-1 left-1 z-10 drop-shadow-[0_1px_2px_rgba(0,0,0,0.6)]">
            <ExternalMediaLink id={item.id} mediaType={isTv ? "tv" : "movie"} />
          </div>
        )}
      </div>

      <div className="flex-1 min-w-0 flex items-start gap-4 sm:gap-5">
        <div className="flex-1 min-w-0">
          <span
            className={`inline-block text-[0.65rem] font-bold uppercase tracking-[0.08em] px-2 py-0.5 rounded-full mb-1.5 ${
              isTv ? "bg-blue-400/10 text-blue-300" : "bg-emerald-400/10 text-emerald-300"
            }`}
          >
            {isTv ? "TV" : "Film"}
          </span>
          <h3 className="text-base font-semibold leading-snug tracking-[-0.02em] mb-0.5 line-clamp-2">
            {title}
          </h3>
          {item.release_date && (
            <p className="text-xs text-muted mb-1 m-0">{item.release_date.slice(0, 4)}</p>
          )}
          {overview && (
            <p className="text-sm text-muted leading-relaxed line-clamp-3 m-0">
              {String(item.overview).slice(0, 380)}
              {String(item.overview).length > 380 ? "…" : ""}
            </p>
          )}
        </div>

        <div className="shrink-0 self-center flex items-center gap-2">
          {!hideWatchlist && item.id && item.media_type && (
            <button
              type="button"
              onClick={() => void toggle({ id: item.id, title: item.title ?? String(item.id), mediaType: item.media_type!, posterPath: item.poster_path, overview: item.overview, releaseDate: item.release_date })}
              disabled={isToggling(item.id, item.media_type)}
              title={isOnWatchlist(item.id, item.media_type) ? "Remove from watchlist" : "Add to watchlist"}
              className={`w-8 h-8 flex items-center justify-center rounded-full transition-all cursor-pointer border font-sans ${
                isOnWatchlist(item.id, item.media_type)
                  ? "bg-accent text-bg border-accent/60 shadow-sm shadow-black/30"
                  : "bg-transparent text-muted border-border hover:text-accent hover:border-accent/40"
              } ${isToggling(item.id, item.media_type) ? "opacity-60" : ""}`}
            >
              {isToggling(item.id, item.media_type) ? (
                <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill={isOnWatchlist(item.id, item.media_type) ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>
                </svg>
              )}
            </button>
          )}
          {!hideWillLike && item.id && item.media_type && (
            <button
              type="button"
              className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-sm font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer whitespace-nowrap"
              disabled={will.status === "loading"}
              onClick={() => void checkWillLike()}
            >
              {will.status === "loading" ? "Checking…" : "Will I Like?"}
            </button>
          )}
        </div>
      </div>

      {willOutput && (
        <div className="w-full flex flex-col gap-1.5">
          {will.status === "ok" && (
            will.data.already_watched ? (
              <span className="text-sm font-semibold text-emerald-300">Already watched</span>
            ) : (
              <>
                <VerdictBadge
                  willLike={Boolean(will.data.will_like)}
                  score={typeof will.data.score === "number" ? Number(will.data.score) : undefined}
                />
                <AiBlurb>{String(will.data.explanation ?? "")}</AiBlurb>
              </>
            )
          )}
          {will.status === "err" && (
            <p className="text-sm text-danger m-0">{will.message}</p>
          )}
        </div>
      )}
    </article>
  );
}
