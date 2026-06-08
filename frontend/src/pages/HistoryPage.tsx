import { ErrorBox } from "../components/ErrorBox";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { apiFetch } from "../api/client";
import {
  apiJson,
  getHistoryQuery,
  pollJobUntil,
  type HistoryRow,
} from "../api/watcher";
import {
  computeWatchTimeMinutes,
  countMoviesShows,
  fmtMinutes,
} from "../lib/historyStats";
import { placeholderPoster, posterUrl } from "../lib/poster";

type MediaFilter = "all" | "movie" | "tv";
type SortKey = "latest" | "earliest" | "title" | "year" | "rewatch_engagement";

const AUTO_SYNC_MS = 300_000;

function rowTitle(row: HistoryRow): string {
  return (row.title as string) || (row.name as string) || String(row.id ?? "—");
}

function rowId(row: HistoryRow): number {
  return Number(row.tmdb_id ?? row.id ?? 0);
}

function fmtDate(iso: string | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (!isNaN(d.getTime()))
      return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch { /* ignore */ }
  return String(iso).slice(0, 10);
}

function sortRows(rows: HistoryRow[], key: SortKey): HistoryRow[] {
  const copy = [...rows];
  copy.sort((a, b) => {
    if (key === "title") return rowTitle(a).localeCompare(rowTitle(b));
    if (key === "rewatch_engagement")
      return (Number(b.rewatch_engagement) || 0) - (Number(a.rewatch_engagement) || 0);
    if (key === "year")
      return (Number(b.year) || 0) - (Number(a.year) || 0);
    if (key === "earliest") {
      const ea = String(a.earliest_watched_at ?? "");
      const eb = String(b.earliest_watched_at ?? "");
      return ea.localeCompare(eb);
    }
    const la = String(a.latest_watched_at ?? "");
    const lb = String(b.latest_watched_at ?? "");
    return lb.localeCompare(la);
  });
  return copy;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-end gap-px">
      <span className="text-[0.8rem] font-semibold text-text/80 tabular-nums">{value}</span>
      <span className="text-[0.6rem] uppercase tracking-[0.06em] text-muted">{label}</span>
    </div>
  );
}

const inputCls = "glass-input rounded-lg text-text px-2.5 py-2 text-[16px] sm:text-sm";

export function HistoryPage() {
  const [media, setMedia] = useState<MediaFilter>("all");
  const [sort, setSort] = useState<SortKey>("latest");
  const [search, setSearch] = useState("");
  const [yearFilter, setYearFilter] = useState("");
  const [genreFilter, setGenreFilter] = useState("");
  const [rows, setRows] = useState<HistoryRow[] | null>(null);
  const [rowsOverall, setRowsOverall] = useState<HistoryRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [syncNote, setSyncNote] = useState<string | null>(null);
  const [autoSyncNote, setAutoSyncNote] = useState<string | null>(null);
  const lastAutoSyncRef = useRef(0);

  const load = useCallback(async () => {
    setErr(null);
    setSyncNote(null);
    try {
      const mt = media === "all" ? null : media;
      const path = getHistoryQuery(mt, true);
      const r = await apiFetch(path);
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as HistoryRow[];
      setRows(data);

      if (media !== "all") {
        const r2 = await apiFetch("/history?include_posters=false");
        if (r2.ok) setRowsOverall((await r2.json()) as HistoryRow[]);
        else setRowsOverall(null);
      } else {
        setRowsOverall(null);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load history");
      setRows(null);
      setRowsOverall(null);
    }
  }, [media]);

  useEffect(() => {
    void load();
  }, [load]);

  const filtered = useMemo(() => {
    if (!rows) return [];
    let r = sortRows(rows, sort);
    const q = search.trim().toLowerCase();
    if (q) r = r.filter((row) => rowTitle(row).toLowerCase().includes(q));
    if (yearFilter) {
      const y = parseInt(yearFilter);
      r = r.filter((row) => {
        const watchedYear = new Date(String(row.latest_watched_at ?? "")).getFullYear();
        return watchedYear === y;
      });
    }
    if (genreFilter) {
      r = r.filter((row) => {
        const genres = (row.genres as string[] | undefined) ?? [];
        return genres.includes(genreFilter);
      });
    }
    return r;
  }, [rows, sort, search, yearFilter, genreFilter]);

  const genreOptions = useMemo(() => {
    if (!rows) return [];
    const genres = new Set<string>();
    for (const row of rows) {
      for (const g of (row.genres as string[] | undefined) ?? []) {
        if (g) genres.add(g);
      }
    }
    return Array.from(genres).sort();
  }, [rows]);

  const watchYears = useMemo(() => {
    if (!rows) return [];
    const years = new Set<number>();
    for (const row of rows) {
      const y = new Date(String(row.latest_watched_at ?? "")).getFullYear();
      if (!isNaN(y)) years.add(y);
    }
    return Array.from(years).sort((a, b) => b - a);
  }, [rows]);

  const statsSource = useMemo(() => {
    if (!rows) return [];
    if (media === "all") return rows;
    return rowsOverall ?? rows;
  }, [rows, rowsOverall, media]);

  const headlineCounts = useMemo(() => countMoviesShows(statsSource), [statsSource]);
  const watchStats = useMemo(() => computeWatchTimeMinutes(statsSource), [statsSource]);

  async function syncNow(showNote = true) {
    setBusy(true);
    if (showNote) setSyncNote(null);
    try {
      const r = await apiFetch("/admin/sync/trakt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!r.ok) {
        if (showNote) setSyncNote(await r.text());
        return;
      }
      const j = (await r.json()) as { job_id?: string };
      if (j.job_id) {
        if (showNote) setSyncNote("Sync running…");
        const done = await pollJobUntil(j.job_id, "trakt", { maxWaitSec: 120 });
        if (done?.status === "completed") {
          if (showNote) setSyncNote("Sync completed.");
          await apiJson("/admin/clear-history-cache", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
          });
          await load();
        } else if (done?.status === "failed") {
          if (showNote) setSyncNote(`Sync failed: ${done.error ?? "unknown"}`);
        } else {
          if (showNote) setSyncNote("Sync still running in the background.");
        }
      } else {
        if (showNote) setSyncNote("Sync accepted.");
        await load();
      }
    } catch (e) {
      if (showNote) setSyncNote(e instanceof Error ? e.message : "Sync error");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    const tick = () => {
      if (document.visibilityState !== "visible") return;
      const now = Date.now();
      if (now - lastAutoSyncRef.current < AUTO_SYNC_MS) return;
      lastAutoSyncRef.current = now;
      void (async () => {
        try {
          const r = await apiFetch("/admin/sync/trakt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          if (r.ok) {
            setAutoSyncNote("Background Trakt sync triggered.");
            setTimeout(() => setAutoSyncNote(null), 4000);
            void load();
          }
        } catch {
          /* optional background sync */
        }
      })();
    };
    const id = window.setInterval(tick, AUTO_SYNC_MS);
    return () => clearInterval(id);
  }, [load]);

  const listCount = filtered.length;
  const totalRows = rows?.length ?? 0;

  return (
    <div className="w-full">
      <h1 className="page-title">Watch History</h1>
      <p className="text-muted mb-6">
        Filter and search your library. With this page open, Trakt is refreshed about every five
        minutes. Use <strong>Sync Trakt Now</strong> for an immediate update.
      </p>

      {/* Stats strip */}
      {rows && rows.length > 0 && (
        <div className="grid grid-cols-3 gap-4 p-5 glass-dark rounded-2xl mb-4">
          {[
            { label: "Movies", count: headlineCounts.movies, time: fmtMinutes(watchStats.movie) },
            { label: "Shows", count: headlineCounts.tv, time: fmtMinutes(watchStats.show) },
            {
              label: "Total Time",
              count: `${(watchStats.total / (60 * 24)).toFixed(1)}d`,
              time: fmtMinutes(watchStats.total),
            },
          ].map(({ label, count, time }) => (
            <div key={label}>
              <h3 className="text-[0.72rem] font-semibold uppercase tracking-[0.06em] text-muted mb-1.5">
                {label}
              </h3>
              <p className="text-[1.65rem] font-bold tracking-tight leading-tight m-0 bg-gradient-to-b from-white to-text/65 bg-clip-text text-transparent">{count}</p>
              <p className="text-xs text-muted mt-0.5">{time} incl. rewatches</p>
            </div>
          ))}
        </div>
      )}

      {/* Toolbar */}
      <div className="p-4 glass-dark rounded-2xl mb-4">
        <div className="flex flex-wrap gap-3 items-end mb-3">
          <label className="flex flex-col gap-1">
            <span className="field-label">
              Media
            </span>
            <select
              className={inputCls}
              value={media}
              onChange={(e) => setMedia(e.target.value as MediaFilter)}
            >
              <option value="all">All</option>
              <option value="movie">Movies</option>
              <option value="tv">TV</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="field-label">
              Sort
            </span>
            <select
              className={inputCls}
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
            >
              <option value="latest">Latest Watched</option>
              <option value="earliest">Earliest Watched</option>
              <option value="year">Release Year</option>
              <option value="title">Title</option>
              <option value="rewatch_engagement">Engagement</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 flex-1 min-w-[180px]">
            <span className="field-label">
              Search
            </span>
            <input
              className={`${inputCls} w-full`}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Title…"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="field-label">
              Genre
            </span>
            <select
              className={inputCls}
              value={genreFilter}
              onChange={(e) => setGenreFilter(e.target.value)}
            >
              <option value="">All</option>
              {genreOptions.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="field-label">
              Watch Year
            </span>
            <select
              className={inputCls}
              value={yearFilter}
              onChange={(e) => setYearFilter(e.target.value)}
            >
              <option value="">All</option>
              {watchYears.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="flex flex-wrap gap-3 items-center">
          <button
            type="button"
            className="btn-primary"
            disabled={busy}
            onClick={() => void syncNow(true)}
          >
            {busy ? "Syncing…" : "Sync Trakt Now"}
          </button>
          <button
            type="button"
            className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-transparent text-muted border border-border font-semibold text-sm cursor-pointer transition-colors hover:text-text hover:border-muted"
            onClick={() => void load()}
          >
            Reload
          </button>
          <span className="text-muted text-sm ml-auto">
            {listCount} of {totalRows} · Movies {headlineCounts.movies} · TV {headlineCounts.tv}
          </span>
        </div>
        {syncNote && <p className="text-sm text-muted mt-2 mb-0">{syncNote}</p>}
        {autoSyncNote && <p className="text-sm text-muted mt-2 mb-0">{autoSyncNote}</p>}
      </div>

      {err && <ErrorBox message={err} />}

      {rows === null && !err && <p className="text-muted">Loading…</p>}
      {rows && rows.length === 0 && (
        <p className="text-muted">No history yet. Sign in and sync Trakt.</p>
      )}

      {filtered.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {filtered.map((row, i) => {
            const title = rowTitle(row);
            const id = rowId(row);
            const mt = String(row.media_type ?? "");
            const mtEnc = encodeURIComponent(mt === "tv" ? "tv" : "movie");
            const src =
              posterUrl(row.poster_path as string | undefined, "w185") ??
              placeholderPoster();
            const isTv = mt === "tv";
            const re = Number(row.rewatch_engagement) || 0;
            return (
              <article
                key={`${id}-${mt}-${i}`}
                className="flex items-center gap-3 px-3 py-2.5 glass glass-hover rounded-xl"
              >
                <div className="w-10 shrink-0 rounded-md overflow-hidden bg-bg shadow">
                  <img src={src} alt="" loading="lazy" className="w-full aspect-[2/3] object-contain block" />
                </div>

                {/* Title + secondary */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span
                      className={`text-[0.6rem] font-bold uppercase tracking-[0.08em] px-1.5 py-0.5 rounded-full shrink-0 ${
                        isTv ? "bg-blue-400/10 text-blue-300" : "bg-emerald-400/10 text-emerald-300"
                      }`}
                    >
                      {isTv ? "TV" : "Film"}
                    </span>
                    <span className="text-sm font-semibold leading-snug tracking-[-0.01em] truncate">
                      {title}
                    </span>
                    {id > 0 && mt && (
                      <>
                        <a
                          className="inline-flex items-center shrink-0 text-muted hover:text-text hover:no-underline transition-colors"
                          href={`https://www.themoviedb.org/${mt === "tv" ? "tv" : "movie"}/${id}`}
                          target="_blank"
                          rel="noreferrer"
                          title="Open in TMDB"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                          </svg>
                        </a>
                      </>
                    )}
                  </div>
                  <p className="text-xs text-muted m-0 pl-[calc(1.5rem+6px)]">
                    {isTv
                      ? `${String(row.watched_episodes ?? 0)} / ${String(row.total_episodes ?? "—")} episodes`
                      : row.year != null ? String(row.year) : "—"}
                    {Array.isArray(row.genres) && (row.genres as string[]).length > 0 && (
                      <span className="ml-2 text-muted/60">
                        {(row.genres as string[]).slice(0, 2).join(" · ")}
                      </span>
                    )}
                  </p>
                </div>

                {/* Stats */}
                <div className="hidden lg:flex items-center gap-4 shrink-0">
                  <Stat
                    label={isTv ? "complete" : "watches"}
                    value={isTv
                      ? `${((Number(row.completion_ratio) || 0) * 100).toFixed(0)}%`
                      : String(row.watch_count ?? 1)}
                  />
                  {re > 0 && <Stat label="rewatch" value={`×${re.toFixed(1)}`} />}
                </div>

                {/* Date */}
                <span className="text-xs text-muted whitespace-nowrap hidden md:block">
                  {fmtDate(String(row.latest_watched_at ?? ""))}
                </span>

                {id > 0 && mt && (
                  <Link
                    className="shrink-0 inline-flex items-center justify-center px-3 py-1.5 rounded-lg text-xs font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 transition-all whitespace-nowrap"
                    to={`/similar?id=${id}&type=${mtEnc}`}
                  >
                    Similar
                  </Link>
                )}
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
