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
type SortKey =
  | "latest"
  | "earliest"
  | "title"
  | "watch_count"
  | "rewatch_engagement";

const AUTO_SYNC_MS = 300_000;

function rowTitle(row: HistoryRow): string {
  return (row.title as string) || (row.name as string) || String(row.id ?? "—");
}

function rowId(row: HistoryRow): number {
  return Number(row.tmdb_id ?? row.id ?? 0);
}

function sortRows(rows: HistoryRow[], key: SortKey): HistoryRow[] {
  const copy = [...rows];
  copy.sort((a, b) => {
    if (key === "title") return rowTitle(a).localeCompare(rowTitle(b));
    if (key === "watch_count")
      return (Number(b.watch_count) || 0) - (Number(a.watch_count) || 0);
    if (key === "rewatch_engagement")
      return (
        (Number(b.rewatch_engagement) || 0) - (Number(a.rewatch_engagement) || 0)
      );
    if (key === "earliest") {
      const ea = String(a.earliest_watched_at ?? "");
      const eb = String(b.earliest_watched_at ?? "");
      return eb.localeCompare(ea);
    }
    const la = String(a.latest_watched_at ?? "");
    const lb = String(b.latest_watched_at ?? "");
    return lb.localeCompare(la);
  });
  return copy;
}

const inputCls =
  "bg-bg border border-border rounded-lg text-text px-2.5 py-2 font-sans text-sm outline-none transition-colors focus:border-accent/50";

export function HistoryPage() {
  const [media, setMedia] = useState<MediaFilter>("all");
  const [sort, setSort] = useState<SortKey>("latest");
  const [search, setSearch] = useState("");
  const [posters, setPosters] = useState(true);
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
      const path = getHistoryQuery(mt, posters);
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
  }, [media, posters]);

  useEffect(() => {
    void load();
  }, [load]);

  const filtered = useMemo(() => {
    if (!rows) return [];
    let r = sortRows(rows, sort);
    const q = search.trim().toLowerCase();
    if (q) r = r.filter((row) => rowTitle(row).toLowerCase().includes(q));
    return r;
  }, [rows, sort, search]);

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
      <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-1.5 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">Watch History</h1>
      <p className="text-muted max-w-[52ch] mb-6">
        Filter and search your library. With this page open, Trakt is refreshed about every five
        minutes. Use <strong>Sync Trakt Now</strong> for an immediate update.
      </p>

      {/* Stats strip */}
      {rows && rows.length > 0 && (
        <div className="grid grid-cols-3 gap-4 p-5 bg-surface border border-border rounded-xl mb-4">
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
      <div className="p-4 bg-surface border border-border rounded-xl mb-4">
        <div className="flex flex-wrap gap-3 items-end mb-3">
          <label className="flex flex-col gap-1">
            <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
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
            <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
              Sort
            </span>
            <select
              className={inputCls}
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
            >
              <option value="latest">Latest Watched</option>
              <option value="earliest">Earliest Watched</option>
              <option value="title">Title</option>
              <option value="watch_count">Watch Count</option>
              <option value="rewatch_engagement">Rewatch Engagement</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 flex-1 min-w-[180px]">
            <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
              Search
            </span>
            <input
              className={`${inputCls} w-full`}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Title…"
            />
          </label>
          <label className="flex items-center gap-2 self-end pb-2">
            <input
              type="checkbox"
              checked={posters}
              onChange={(e) => setPosters(e.target.checked)}
            />
            <span className="text-sm">Poster Cards</span>
          </label>
        </div>
        <div className="flex flex-wrap gap-3 items-center">
          <button
            type="button"
            className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-white font-semibold text-sm cursor-pointer transition-all hover:brightness-110 hover:shadow-[0_0_20px_-4px] hover:shadow-accent/40 disabled:opacity-50 disabled:cursor-not-allowed border-0"
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

      {err && (
        <div className="p-4 bg-surface border border-danger/40 rounded-xl mb-4">
          <strong className="text-danger">Error: </strong>
          {err}
        </div>
      )}

      {rows === null && !err && <p className="text-muted">Loading…</p>}
      {rows && rows.length === 0 && (
        <p className="text-muted">No history yet. Sign in and sync Trakt.</p>
      )}

      {/* Poster card list */}
      {filtered.length > 0 && posters && (
        <div className="flex flex-col gap-3">
          {filtered.map((row, i) => {
            const title = rowTitle(row);
            const id = rowId(row);
            const mt = String(row.media_type ?? "");
            const mtEnc = encodeURIComponent(mt === "tv" ? "tv" : "movie");
            const src =
              posterUrl(row.poster_path as string | undefined, "w185") ??
              placeholderPoster(title);
            const re = Number(row.rewatch_engagement) || 0;
            const isTv = mt === "tv";
            return (
              <article
                key={`${id}-${mt}-${i}`}
                className="flex gap-4 sm:gap-5 p-4 sm:p-5 bg-surface border border-border rounded-xl transition-all duration-200 hover:border-accent/30 hover:shadow-[0_8px_30px_-8px] hover:shadow-accent/20"
              >
                <div className="w-[64px] sm:w-[76px] shrink-0 rounded-lg overflow-hidden bg-bg shadow-md shadow-black/30">
                  <img src={src} alt="" loading="lazy" className="w-full aspect-[2/3] object-cover block" />
                </div>
                <div className="flex-1 min-w-0">
                  <span
                    className={`inline-block text-[0.65rem] font-bold uppercase tracking-[0.08em] px-2 py-0.5 rounded-full mb-1.5 ${
                      isTv ? "bg-blue-400/10 text-blue-300" : "bg-emerald-400/10 text-emerald-300"
                    }`}
                  >
                    {isTv ? "TV" : "Film"}
                  </span>
                  <h3 className="text-base font-semibold leading-snug tracking-[-0.02em] mb-1 line-clamp-2">
                    {title}
                  </h3>
                  {isTv ? (
                    <p className="text-sm text-muted m-0">
                      {String(row.watched_episodes ?? 0)} /{" "}
                      {String(row.total_episodes ?? "—")} episodes
                    </p>
                  ) : (
                    <p className="text-sm text-muted m-0">
                      {row.year != null ? String(row.year) : "—"}
                    </p>
                  )}
                </div>
                <ul className="hidden sm:flex flex-col gap-1.5 text-xs shrink-0 w-28 list-none m-0 p-0">
                  {isTv ? (
                    <li className="flex justify-between items-baseline gap-2">
                      <span className="text-[0.72rem] uppercase tracking-[0.04em] text-muted">Complete</span>
                      <span className="font-semibold">
                        {`${((Number(row.completion_ratio) || 0) * 100).toFixed(0)}%`}
                      </span>
                    </li>
                  ) : (
                    <li className="flex justify-between items-baseline gap-2">
                      <span className="text-[0.72rem] uppercase tracking-[0.04em] text-muted">Watches</span>
                      <span className="font-semibold">{String(row.watch_count ?? 0)}</span>
                    </li>
                  )}
                  {re > 0 && (
                    <li className="flex justify-between items-baseline gap-2">
                      <span className="text-[0.72rem] uppercase tracking-[0.04em] text-muted">Rewatch</span>
                      <span className="font-semibold">×{re.toFixed(1)}</span>
                    </li>
                  )}
                  <li className="flex justify-between items-baseline gap-2">
                    <span className="text-[0.72rem] uppercase tracking-[0.04em] text-muted">Last</span>
                    <span className="font-semibold font-mono text-[0.78rem]">
                      {String(row.latest_watched_at ?? "—").slice(0, 10)}
                    </span>
                  </li>
                </ul>
                <div className="shrink-0 self-center">
                  {id > 0 && mt && (
                    <Link
                      className="inline-flex items-center justify-center px-3 py-2 rounded-lg text-sm font-semibold bg-accent/10 text-accent border border-accent/25 hover:bg-accent/15 hover:border-accent/40 hover:no-underline transition-all whitespace-nowrap"
                      to={`/similar?id=${id}&type=${mtEnc}`}
                    >
                      Similar
                    </Link>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}

      {/* Table view */}
      {filtered.length > 0 && !posters && (
        <div className="overflow-x-auto border border-border rounded-xl bg-surface">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                {["Title", "Type", "Latest", "Count", ""].map((h) => (
                  <th
                    key={h}
                    className="text-left px-4 py-3 text-[0.72rem] uppercase tracking-[0.05em] text-muted font-semibold border-b border-border"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => {
                const id = rowId(row);
                const mt = String(row.media_type ?? "");
                const mtEnc = encodeURIComponent(mt === "tv" ? "tv" : "movie");
                return (
                  <tr
                    key={`${row.id}-${row.media_type}-${i}`}
                    className="border-b border-border last:border-b-0 hover:bg-accent/5 transition-colors"
                  >
                    <td className="px-4 py-3">{rowTitle(row)}</td>
                    <td className="px-4 py-3">{mt || "—"}</td>
                    <td className="px-4 py-3 font-mono text-[0.85em]">
                      {String(row.latest_watched_at ?? "—")}
                    </td>
                    <td className="px-4 py-3">{String(row.watch_count ?? "—")}</td>
                    <td className="px-4 py-3">
                      {id > 0 && mt ? (
                        <Link to={`/similar?id=${id}&type=${mtEnc}`}>Similar</Link>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
