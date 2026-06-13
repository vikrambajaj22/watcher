import { useMemo, useState } from "react";
import { MediaCard } from "../components/MediaCard";
import { SearchTypeahead } from "../components/SearchTypeahead";
import { ErrorBox } from "../components/ErrorBox";
import { type SearchHit } from "../api/watcher";
import { useWatchlist } from "../contexts/WatchlistContext";

type Filter = "all" | "movie" | "tv";

export function WatchlistPage() {
  const { watchlist, toggle, sync, isOnWatchlist, isToggling } = useWatchlist();
  const [filter, setFilter] = useState<Filter>("all");
  const [activeGenres, setActiveGenres] = useState<Set<string>>(new Set());
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<{ added: number; removed: number } | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [adding, setAdding] = useState(false);

  const byType = useMemo(() => {
    const filtered = filter === "all" ? watchlist : watchlist.filter((w) => w.media_type === filter);
    return filtered;
  }, [watchlist, filter]);

  const allGenres = useMemo(() => {
    const counts: Record<string, number> = {};
    byType.forEach((w) => (w.genres ?? []).forEach((g) => { counts[g] = (counts[g] ?? 0) + 1; }));
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([g]) => g);
  }, [byType]);

  const visible = useMemo(() => {
    if (activeGenres.size === 0) return byType;
    return byType.filter((w) => (w.genres ?? []).some((g) => activeGenres.has(g)));
  }, [byType, activeGenres]);

  const movieCount = watchlist.filter((w) => w.media_type === "movie").length;
  const tvCount = watchlist.filter((w) => w.media_type === "tv").length;

  function toggleGenre(g: string) {
    setActiveGenres((prev) => {
      const next = new Set(prev);
      if (next.has(g)) next.delete(g); else next.add(g);
      return next;
    });
  }

  async function handleSync() {
    setSyncing(true);
    setSyncError(null);
    setSyncResult(null);
    try {
      setSyncResult(await sync());
    } catch {
      setSyncError("Sync failed — check Trakt connection.");
    } finally {
      setSyncing(false);
    }
  }

  async function handleAdd() {
    if (!selectedHit) return;
    setAdding(true);
    try {
      await toggle({
        id: selectedHit.id,
        title: selectedHit.title,
        mediaType: selectedHit.media_type,
        posterPath: selectedHit.poster_path,
      });
      setSelectedHit(null);
    } finally {
      setAdding(false);
    }
  }

  const tabs: { label: string; value: Filter; count: number }[] = [
    { label: "All", value: "all", count: watchlist.length },
    { label: "Movies", value: "movie", count: movieCount },
    { label: "TV Shows", value: "tv", count: tvCount },
  ];

  return (
    <div className="w-full">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-6">
        <div>
          <h1 className="page-title mb-1">Watchlist</h1>
          <p className="text-muted text-sm m-0">
            {movieCount} movie{movieCount !== 1 ? "s" : ""} · {tvCount} show{tvCount !== 1 ? "s" : ""}
            {" "}synced with Trakt
          </p>
        </div>
        <button
          type="button"
          className="px-4 py-2 rounded-lg text-sm font-semibold glass border border-border text-muted hover:text-text hover:border-white/20 transition-all cursor-pointer font-sans bg-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={syncing}
          onClick={() => void handleSync()}
        >
          {syncing ? "Syncing…" : "Sync from Trakt"}
        </button>
      </div>

      {syncError && <ErrorBox message={syncError} />}
      {syncResult && (
        <p className="text-sm text-accent mb-4">
          Synced — {syncResult.added} added, {syncResult.removed} removed.
        </p>
      )}

      {/* Add via search */}
      <div className="p-4 glass-dark rounded-2xl mb-5 flex flex-wrap items-end gap-3">
        <div className="flex-1 min-w-[220px] flex flex-col gap-1">
          <span className="field-label">Add title</span>
          <SearchTypeahead
            selected={selectedHit}
            onSelect={(hit) => setSelectedHit(hit)}
            onClear={() => setSelectedHit(null)}
            placeholder="Search movie or show…"
          />
        </div>
        <button
          type="button"
          className="btn-primary"
          disabled={
            !selectedHit ||
            adding ||
            (selectedHit != null && isOnWatchlist(selectedHit.id, selectedHit.media_type))
          }
          onClick={() => void handleAdd()}
        >
          {adding
            ? "Adding…"
            : selectedHit && isOnWatchlist(selectedHit.id, selectedHit.media_type)
            ? "Already on list"
            : "Add"}
        </button>
      </div>

      {/* Type + genre filters */}
      <div className="flex gap-2 mb-3 flex-wrap">
        {tabs.map(({ label, value, count }) => (
          <button
            key={value}
            type="button"
            onClick={() => { setFilter(value); setActiveGenres(new Set()); }}
            className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all cursor-pointer border font-sans ${
              filter === value
                ? "bg-accent/15 text-accent border-accent/35"
                : "text-muted border-border hover:text-text hover:border-muted bg-transparent"
            }`}
          >
            {label}
            <span className="ml-1.5 text-xs opacity-70">{count}</span>
          </button>
        ))}
      </div>

      {allGenres.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-5">
          {allGenres.map((g) => {
            const on = activeGenres.has(g);
            return (
              <button
                key={g}
                type="button"
                onClick={() => toggleGenre(g)}
                className={`px-2.5 py-0.5 rounded-full text-xs font-medium transition-all cursor-pointer border font-sans ${
                  on
                    ? "bg-accent/20 text-accent border-accent/40"
                    : "bg-transparent text-muted border-border hover:text-text hover:border-muted"
                }`}
              >
                {g}
              </button>
            );
          })}
          {activeGenres.size > 0 && (
            <button
              type="button"
              onClick={() => setActiveGenres(new Set())}
              className="px-2.5 py-0.5 rounded-full text-xs font-medium text-muted border border-transparent hover:text-text transition-all cursor-pointer bg-transparent font-sans"
            >
              Clear
            </button>
          )}
        </div>
      )}

      {visible.length === 0 ? (
        <p className="empty-state">
          {watchlist.length === 0
            ? "Your watchlist is empty. Add titles above or sync from Trakt."
            : "No items match the selected filters."}
        </p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {visible.map((item) => (
            <MediaCard
              key={`${item.tmdb_id}-${item.media_type}`}
              id={item.tmdb_id}
              title={item.title ?? "Unknown"}
              mediaType={item.media_type}
              posterPath={item.poster_path}
              overview={item.overview}
              similarLink
              watchlistOn={true}
              watchlistLoading={isToggling(item.tmdb_id, item.media_type)}
              onWatchlistToggle={() =>
                void toggle({
                  id: item.tmdb_id,
                  title: item.title ?? "",
                  mediaType: item.media_type,
                  posterPath: item.poster_path,
                })
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}
