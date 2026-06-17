import { LoadingBox } from "../components/LoadingBox";
import { useState } from "react";
import { Link } from "react-router-dom";
import { ErrorBox } from "../components/ErrorBox";
import { apiFetch } from "../api/client";
import { type SearchHit, type WillLikeResponse } from "../api/watcher";
import { AiBlurb } from "../components/AiBlurb";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { VerdictBadge } from "../components/VerdictBadge";
import { posterUrl, placeholderPoster } from "../lib/poster";
import { useWatchlist } from "../contexts/WatchlistContext";

export function WillLikePage() {
  const { isOnWatchlist, toggle, isToggling } = useWatchlist();
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
    <div className="w-full">
      <h1 className="page-title">Will I Like It?</h1>
      <p className="text-muted mb-6">
        A short model-assisted read on whether a title fits your taste, based on your watch history.
      </p>

      <div className="relative z-10 p-5 glass-dark rounded-2xl mb-4">
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="field-label">
            Title
          </span>
          <SearchTypeahead
            selected={selectedHit}
            onSelect={(hit) => { setSelectedHit(hit); setResult(null); setErr(null); }}
            onClear={() => setSelectedHit(null)}
            placeholder="Search movie or show…"
          />
        </label>
        <button
          type="button"
          className="btn-primary"
          disabled={busy || !selectedHit}
          onClick={() => void submit()}
        >
          {busy ? "Checking…" : "Check"}
        </button>
      </div>

      {err && <ErrorBox message={err} />}

      {busy && (
        <LoadingBox label="Checking Your Taste Fit…" />
      )}

      {!busy && result && (
        <article className="flex flex-wrap items-start gap-4 sm:gap-5 p-4 sm:p-5 glass rounded-2xl">
          <div className="w-[64px] sm:w-[76px] shrink-0 rounded-lg overflow-hidden bg-bg shadow-md shadow-black/30">
            <img
              src={posterUrl(result.item.poster_path, "w185") ?? placeholderPoster()}
              alt=""
              loading="lazy"
              className="w-full aspect-[2/3] object-contain block"
            />
          </div>

          <div className="flex-1 min-w-0 flex items-start gap-4 sm:gap-5">
            <div className="flex-1 min-w-0">
              <span
                className={`inline-block text-[0.65rem] font-bold uppercase tracking-[0.08em] px-2 py-0.5 rounded-full mb-1.5 ${
                  result.already_watched
                    ? "bg-emerald-400/10 text-emerald-300"
                    : isTv ? "bg-blue-400/10 text-blue-300" : "bg-emerald-400/10 text-emerald-300"
                }`}
              >
                {result.already_watched ? "Watched" : isTv ? "TV" : "Film"}
              </span>
              <h2 className="text-base font-semibold leading-snug tracking-[-0.02em] mb-0.5 flex items-center gap-1.5">
                {displayTitle}
                {itemId != null && itemId > 0 && (
                  <a
                    className="inline-flex items-center text-muted hover:text-text hover:no-underline transition-colors shrink-0"
                    href={`https://www.themoviedb.org/${itemMt}/${itemId}`}
                    target="_blank"
                    rel="noreferrer"
                    title="Open in TMDB"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                    </svg>
                  </a>
                )}
              </h2>
            </div>

            <div className="shrink-0 self-start">
              {itemId != null && itemId > 0 && result.item.media_type && (
                <Link
                  className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-sm font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 transition-all whitespace-nowrap"
                  to={`/similar?id=${itemId}&type=${encodeURIComponent(itemMt)}`}
                >
                  Find Similar
                </Link>
              )}
            </div>
          </div>

          <div className="w-full flex flex-col gap-2">
            {!result.already_watched && <VerdictBadge willLike={result.will_like} score={result.score} />}
            {result.already_watched && <p className="text-sm text-muted m-0">Already in your history.</p>}
            <AiBlurb>{result.explanation}</AiBlurb>
            {!result.already_watched && itemId != null && itemId > 0 && result.item.media_type && (
              <button
                type="button"
                onClick={() => void toggle({ id: itemId, title: displayTitle, mediaType: itemMt, posterPath: result.item.poster_path })}
                disabled={isToggling(itemId, itemMt)}
                className={`self-start inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-semibold border transition-all cursor-pointer font-sans ${
                  isOnWatchlist(itemId, itemMt)
                    ? "bg-accent/15 text-accent border-accent/35 hover:bg-accent/10"
                    : "bg-transparent text-muted border-border hover:text-text hover:border-white/25"
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isToggling(itemId, itemMt) ? (
                  <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill={isOnWatchlist(itemId, itemMt) ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>
                  </svg>
                )}
                {isOnWatchlist(itemId, itemMt) ? "On your watchlist" : "Add to watchlist"}
              </button>
            )}
          </div>
        </article>
      )}
    </div>
  );
}
