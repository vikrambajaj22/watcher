import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { AiBlurb } from "./AiBlurb";
import { ExternalMediaLink } from "./ExternalMediaLink";
import { placeholderPoster, posterUrl } from "../lib/poster";

type Props = {
  id: number;
  title: string;
  mediaType?: string;
  posterPath?: string | null;
  subtitle?: string;
  overview?: string | null;
  footer?: ReactNode;
  similarLink?: boolean;
  watched?: boolean;
  compact?: boolean;
  watchlistOn?: boolean;
  watchlistLoading?: boolean;
  onWatchlistToggle?: () => void;
  linkTo?: "tmdb" | "trakt";
  traktSlug?: string | null;
};

export function MediaCard({
  id,
  title,
  mediaType,
  posterPath,
  subtitle,
  overview,
  footer,
  similarLink,
  watched,
  compact,
  watchlistOn,
  watchlistLoading,
  onWatchlistToggle,
  linkTo = "tmdb",
  traktSlug,
}: Props) {
  const src = posterUrl(posterPath ?? null, compact ? "w185" : "w342") ?? placeholderPoster();
  const mt = (mediaType || "movie").toLowerCase();
  const isTv = mt === "tv";

  return (
    <article className="glass glass-hover rounded-xl overflow-hidden flex flex-col h-full">
      <div className="aspect-[2/3] bg-bg relative">
        <img className="w-full h-full object-contain block" src={src} alt="" loading="lazy" />
        {id > 0 && (
          <div className="absolute top-1.5 left-1.5 z-10 drop-shadow-[0_1px_2px_rgba(0,0,0,0.6)]">
            <ExternalMediaLink id={id} mediaType={mt} linkTo={linkTo} traktSlug={traktSlug} />
          </div>
        )}
        {onWatchlistToggle && (
          <button
            type="button"
            onClick={(e) => { e.preventDefault(); e.stopPropagation(); onWatchlistToggle(); }}
            disabled={watchlistLoading}
            title={watchlistOn ? "Remove from watchlist" : "Add to watchlist"}
            className={`absolute top-1.5 right-1.5 w-7 h-7 flex items-center justify-center rounded-full transition-all cursor-pointer border font-sans ${
              watchlistOn
                ? "bg-accent text-bg border-accent/60 shadow-md shadow-black/40"
                : "bg-bg/80 text-muted border-white/15 hover:text-accent hover:border-accent/40"
            } ${watchlistLoading ? "opacity-60" : ""}`}
          >
            {watchlistLoading ? (
              <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill={watchlistOn ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>
              </svg>
            )}
          </button>
        )}
      </div>
      <div className={`${compact ? "p-2" : "p-4"} flex-1 flex flex-col gap-1.5`}>
        {(mediaType || watched) && (
          <div className="flex flex-wrap gap-1">
            {mediaType && (
              <span
                className={`text-[0.6rem] font-bold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded-full ${
                  isTv
                    ? "bg-blue-400/10 text-blue-300"
                    : "bg-emerald-400/10 text-emerald-300"
                }`}
              >
                {isTv ? "TV" : "Film"}
              </span>
            )}
            {watched && (
              <span className="text-[0.6rem] font-bold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded-full bg-accent/10 text-accent border border-accent/20">
                Watched
              </span>
            )}
          </div>
        )}
        <h3 className={`${compact ? "text-xs" : "text-base"} font-semibold leading-snug tracking-[-0.02em] m-0`}>
          <span className="line-clamp-2">{title}</span>
        </h3>
        {subtitle && <AiBlurb>{subtitle}</AiBlurb>}
        {!compact && overview && String(overview).trim() && (
          <p className="text-sm text-muted leading-relaxed m-0 line-clamp-4">
            {String(overview).slice(0, 280)}
            {String(overview).length > 280 ? "…" : ""}
          </p>
        )}
        {(footer || similarLink) && (
          <div className="mt-auto flex flex-col gap-1.5 pt-1.5">
            {footer ?? null}
            {similarLink && (
              <Link
                className={`self-stretch text-center ${compact ? "px-2 py-1 text-xs" : "px-3 py-2 text-sm"} rounded-lg font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 hover:no-underline transition-all`}
                to={`/similar?id=${id}&type=${encodeURIComponent(mt)}`}
              >
                Find Similar
              </Link>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
