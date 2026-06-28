import { useEffect, useMemo, useState } from "react";
import { MediaCard } from "../components/MediaCard";
import { ErrorBox } from "../components/ErrorBox";
import { apiJson, type HistoryRow, type UpcomingEpisode, type UpcomingResponse } from "../api/watcher";
import { posterUrl, placeholderPoster } from "../lib/poster";

function rowTitle(row: HistoryRow): string {
  return String(row.title ?? row.name ?? "Untitled");
}

function epCode(season?: number | null, episode?: number | null): string {
  const s = season != null ? `S${String(season).padStart(2, "0")}` : "";
  const e = episode != null ? `E${String(episode).padStart(2, "0")}` : "";
  return `${s}${e}`;
}

function dayLabel(iso: string): string {
  const d = new Date(iso);
  const today = new Date();
  const tomorrow = new Date();
  tomorrow.setDate(today.getDate() + 1);
  const same = (a: Date, b: Date) => a.toDateString() === b.toDateString();
  if (same(d, today)) return "Today";
  if (same(d, tomorrow)) return "Tomorrow";
  return d.toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" });
}

const PAGE_SIZE = 10;

export function WatchingPage() {
  const [inProgress, setInProgress] = useState<HistoryRow[]>([]);
  const [upcoming, setUpcoming] = useState<UpcomingEpisode[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    let active = true;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [prog, cal] = await Promise.all([
          apiJson<HistoryRow[]>("/history/in-progress"),
          apiJson<UpcomingResponse>("/calendar/upcoming"),
        ]);
        if (!active) return;
        setInProgress(prog);
        setUpcoming(cal.episodes);
      } catch {
        if (active) setError("Couldn't load your shows — check your Trakt connection.");
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  // next air date per show, to surface on the in-progress cards
  const nextAirByShow = useMemo(() => {
    const map = new Map<number, UpcomingEpisode>();
    for (const ep of upcoming) {
      if (!map.has(ep.tmdb_id)) map.set(ep.tmdb_id, ep);
    }
    return map;
  }, [upcoming]);

  const grouped = useMemo(() => {
    const groups: { label: string; items: UpcomingEpisode[] }[] = [];
    const order = new Map<string, number>();
    for (const ep of upcoming) {
      if (!ep.first_aired) continue;
      const label = dayLabel(ep.first_aired);
      if (!order.has(label)) {
        order.set(label, groups.length);
        groups.push({ label, items: [] });
      }
      groups[order.get(label)!].items.push(ep);
    }
    return groups;
  }, [upcoming]);

  const pageCount = Math.ceil(inProgress.length / PAGE_SIZE);
  const pageItems = useMemo(
    () => inProgress.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE),
    [inProgress, page],
  );

  return (
    <div className="w-full">
      <div className="mb-6">
        <h1 className="page-title mb-1">Currently Watching</h1>
        <p className="text-muted text-sm m-0">
          Shows you've started but haven't finished, and what's airing next.
        </p>
      </div>

      {error && <ErrorBox message={error} />}

      {loading ? (
        <p className="text-muted text-sm">Loading…</p>
      ) : (
        <>
          <section className="mb-10">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-semibold">
                Up Next{inProgress.length > 0 ? ` (${inProgress.length})` : ""}
              </h2>
            </div>
            {inProgress.length === 0 ? (
              <p className="text-muted text-sm">No shows in progress right now.</p>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {pageItems.map((row) => {
                  const id = Number(row.id);
                  const watched = Number(row.watched_episodes ?? 0);
                  const total = row.total_episodes != null ? Number(row.total_episodes) : null;
                  const pct = Math.round((Number(row.completion_ratio) || 0) * 100);
                  const next = nextAirByShow.get(id);
                  return (
                    <MediaCard
                      key={`${id}-tv`}
                      id={id}
                      title={rowTitle(row)}
                      mediaType="tv"
                      posterPath={row.poster_path as string | undefined}
                      subtitle={`${watched} / ${total ?? "—"} episodes · ${pct}%`}
                      similarLink
                      linkTo="trakt"
                      traktSlug={(row.ids as { slug?: string } | undefined)?.slug}
                      footer={
                        next?.first_aired ? (
                          <span className="text-xs text-accent">
                            Next: {epCode(next.season, next.episode)} · {dayLabel(next.first_aired)}
                          </span>
                        ) : null
                      }
                    />
                  );
                })}
              </div>
            )}
            {pageCount > 1 && (
              <div className="flex items-center justify-center gap-3 mt-5">
                <button
                  type="button"
                  className="px-3 py-1.5 rounded-lg text-sm glass border border-border text-muted hover:text-text hover:border-white/20 transition-all cursor-pointer bg-transparent disabled:opacity-40 disabled:cursor-not-allowed"
                  disabled={page === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                >
                  Prev
                </button>
                <span className="text-sm text-muted">
                  {page + 1} / {pageCount}
                </span>
                <button
                  type="button"
                  className="px-3 py-1.5 rounded-lg text-sm glass border border-border text-muted hover:text-text hover:border-white/20 transition-all cursor-pointer bg-transparent disabled:opacity-40 disabled:cursor-not-allowed"
                  disabled={page >= pageCount - 1}
                  onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                >
                  Next
                </button>
              </div>
            )}
          </section>

          <section>
            <h2 className="text-lg font-semibold mb-3">Upcoming Calendar</h2>
            {grouped.length === 0 ? (
              <p className="text-muted text-sm">No episodes airing in the next two weeks.</p>
            ) : (
              <div className="flex flex-col gap-6">
                {grouped.map((g) => (
                  <div key={g.label}>
                    <h3 className="text-sm font-semibold text-muted mb-2">{g.label}</h3>
                    <div className="flex flex-col gap-2">
                      {g.items.map((ep, i) => (
                        <div
                          key={`${ep.tmdb_id}-${ep.season}-${ep.episode}-${i}`}
                          className="flex items-center gap-3 p-2 glass rounded-lg"
                        >
                          <img
                            className="w-10 h-15 object-cover rounded shrink-0"
                            src={posterUrl(ep.poster_path ?? null, "w185") ?? placeholderPoster()}
                            alt=""
                            loading="lazy"
                          />
                          <div className="min-w-0">
                            <div className="text-sm font-medium truncate">{ep.show_title}</div>
                            <div className="text-xs text-muted truncate">
                              {epCode(ep.season, ep.episode)}
                              {ep.episode_title ? ` · ${ep.episode_title}` : ""}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
