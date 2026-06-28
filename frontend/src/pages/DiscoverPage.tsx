import { LoadingBox } from "../components/LoadingBox";
import { ErrorBox } from "../components/ErrorBox";
import { useState } from "react";
import { apiFetch } from "../api/client";
import { type DescribeFilters, type DiscoverItem, type WillLikeResponse } from "../api/watcher";
import { MediaCard } from "../components/MediaCard";
import { VerdictBadge } from "../components/VerdictBadge";
import { AiBlurb } from "../components/AiBlurb";
import { useWatchlist } from "../contexts/WatchlistContext";

type MediaTypeFilter = "auto" | "both" | "movie" | "tv";

type WillLikeState =
  | { status: "loading" }
  | { status: "done"; data: WillLikeResponse }
  | { status: "error"; message: string };

const cardKey = (item: DiscoverItem) => `${item.id}-${item.media_type ?? ""}`;

export function DiscoverPage() {
  const { isOnWatchlist, toggle, isToggling } = useWatchlist();
  const [query, setQuery] = useState("");
  const [mediaType, setMediaType] = useState<MediaTypeFilter>("auto");
  const [results, setResults] = useState<DiscoverItem[] | null>(null);
  const [filters, setFilters] = useState<DescribeFilters | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [hideWatched, setHideWatched] = useState(false);
  const [willLike, setWillLike] = useState<Record<string, WillLikeState>>({});

  async function checkWillLike(item: DiscoverItem) {
    if (!item.media_type) return;
    const key = cardKey(item);
    setWillLike((prev) => ({ ...prev, [key]: { status: "loading" } }));
    try {
      const r = await apiFetch("/will-like", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tmdb_id: item.id, media_type: item.media_type }),
      });
      const text = await r.text();
      let j: unknown;
      try { j = JSON.parse(text); } catch { j = { detail: text }; }
      if (!r.ok) {
        const message =
          typeof j === "object" && j && "detail" in j
            ? String((j as { detail: unknown }).detail)
            : text;
        setWillLike((prev) => ({ ...prev, [key]: { status: "error", message } }));
        return;
      }
      setWillLike((prev) => ({ ...prev, [key]: { status: "done", data: j as WillLikeResponse } }));
    } catch (e) {
      setWillLike((prev) => ({ ...prev, [key]: { status: "error", message: e instanceof Error ? e.message : "Request failed" } }));
    }
  }

  async function search() {
    const q = query.trim();
    if (!q) return;
    setBusy(true);
    setErr(null);
    setResults(null);
    setFilters(null);
    setWillLike({});
    try {
      const r = await apiFetch("/discover/describe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, limit: 20, media_type: mediaType === "auto" ? null : mediaType }),
      });
      const raw = await r.text();
      let j: unknown;
      try { j = JSON.parse(raw); } catch { j = { detail: raw }; }
      if (!r.ok) {
        setErr(
          typeof j === "object" && j && "detail" in j
            ? String((j as { detail: unknown }).detail)
            : raw,
        );
        return;
      }
      const resp = j as { results: DiscoverItem[]; filters?: DescribeFilters };
      setResults(resp.results ?? []);
      setFilters(resp.filters ?? null);
      // When left on Auto, reflect the media type the backend inferred.
      const inferred = resp.filters?.media_type;
      if (mediaType === "auto" && (inferred === "movie" || inferred === "tv" || inferred === "both")) {
        setMediaType(inferred);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  const visibleResults = (results ?? []).filter((item) => !hideWatched || !item.watched);
  const watchedCount = (results ?? []).filter((item) => item.watched).length;

  const filterChips: string[] = [];
  if (filters) {
    if (filters.genres?.length) filterChips.push(...filters.genres);
    if (filters.cast?.length) filterChips.push(...filters.cast.map((n) => `with ${n}`));
    if (filters.keywords?.length) filterChips.push(...filters.keywords);
    if (filters.year_from && filters.year_to) filterChips.push(`${filters.year_from}–${filters.year_to}`);
    else if (filters.year_from) filterChips.push(`from ${filters.year_from}`);
    else if (filters.year_to) filterChips.push(`up to ${filters.year_to}`);
  }

  return (
    <div className="w-full">
      <h1 className="page-title">
        Discover
      </h1>
      <p className="text-muted mb-6">
        Describe what you want to watch — mood, genre, era, cast — and we'll find it.
      </p>

      <div className="p-5 glass-dark rounded-2xl mb-4">
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="field-label">
            What are you in the mood for?
          </span>
          <input
            className="glass-input rounded-lg text-text px-3 py-2.5 text-[16px] sm:text-sm w-full"
            type="text"
            placeholder="e.g. 90s sci-fi with practical effects, feel-good comedy like The Grand Budapest Hotel…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void search(); }}
          />
        </label>
        <label className="flex flex-col gap-1.5 mb-4">
          <span className="field-label">
            Type
          </span>
          <select
            className="glass-input rounded-lg text-text px-3 py-2.5 text-[16px] sm:text-sm w-36"
            value={mediaType}
            onChange={(e) => setMediaType(e.target.value as MediaTypeFilter)}
          >
            <option value="auto">Auto</option>
            <option value="both">Movies & TV</option>
            <option value="movie">Movies only</option>
            <option value="tv">TV only</option>
          </select>
        </label>
        <button
          type="button"
          className="btn-primary"
          disabled={busy || !query.trim()}
          onClick={() => void search()}
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </div>

      {err && <ErrorBox message={err} />}

      {busy && (
        <LoadingBox label="Searching…" />
      )}

      {!busy && results !== null && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            {filterChips.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {filterChips.map((chip) => (
                  <span
                    key={chip}
                    className="text-[0.7rem] font-semibold uppercase tracking-[0.06em] px-2 py-0.5 rounded-full bg-accent/10 text-accent border border-accent/20"
                  >
                    {chip}
                  </span>
                ))}
              </div>
            ) : <span />}
            {watchedCount > 0 && (
              <label className="flex items-center gap-2 text-sm text-muted cursor-pointer select-none shrink-0">
                <input
                  type="checkbox"
                  className="accent-accent"
                  checked={hideWatched}
                  onChange={(e) => setHideWatched(e.target.checked)}
                />
                Hide watched ({watchedCount})
              </label>
            )}
          </div>
          {visibleResults.length === 0 ? (
            <p className="empty-state">
              {results.length === 0 ? "No results found. Try a different description." : "All results are already watched."}
            </p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {visibleResults.map((item) => {
                const wl = item.media_type ? willLike[cardKey(item)] : undefined;
                return (
                  <MediaCard
                    key={`${item.id}-${item.media_type}`}
                    id={item.id}
                    title={item.title ?? "Unknown"}
                    mediaType={item.media_type}
                    posterPath={item.poster_path}
                    overview={item.overview}
                    watched={item.watched}
                    similarLink
                    footer={
                      item.media_type && !item.watched ? (
                        <WillLikeFooter state={wl} onCheck={() => void checkWillLike(item)} />
                      ) : undefined
                    }
                    watchlistOn={item.media_type ? isOnWatchlist(item.id, item.media_type) : undefined}
                    watchlistLoading={item.media_type ? isToggling(item.id, item.media_type) : undefined}
                    onWatchlistToggle={
                      item.media_type
                        ? () => void toggle({ id: item.id, title: item.title ?? "", mediaType: item.media_type!, posterPath: item.poster_path, overview: item.overview, releaseDate: item.release_date })
                        : undefined
                    }
                  />
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function WillLikeFooter({ state, onCheck }: { state?: WillLikeState; onCheck: () => void }) {
  if (!state) {
    return (
      <button
        type="button"
        onClick={onCheck}
        className="self-stretch text-center px-2 py-1 text-xs rounded-lg font-semibold bg-white/[0.04] text-muted border border-border hover:text-text hover:border-white/25 transition-all cursor-pointer font-sans"
      >
        Will I like it?
      </button>
    );
  }
  if (state.status === "loading") {
    return <p className="text-xs text-muted text-center py-1">Checking your taste…</p>;
  }
  if (state.status === "error") {
    return <p className="text-xs text-danger py-1">{state.message}</p>;
  }
  return (
    <div className="flex flex-col gap-1.5">
      <VerdictBadge willLike={state.data.will_like} score={state.data.score} />
      <AiBlurb>{state.data.explanation}</AiBlurb>
    </div>
  );
}
