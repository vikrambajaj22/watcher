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
  return (
    (row.title as string) ||
    (row.name as string) ||
    String(row.id ?? "—")
  );
}

function rowId(row: HistoryRow): number {
  return Number(row.tmdb_id ?? row.id ?? 0);
}

function sortRows(rows: HistoryRow[], key: SortKey): HistoryRow[] {
  const copy = [...rows];
  copy.sort((a, b) => {
    if (key === "title") {
      return rowTitle(a).localeCompare(rowTitle(b));
    }
    if (key === "watch_count") {
      return (
        (Number(b.watch_count) || 0) - (Number(a.watch_count) || 0)
      );
    }
    if (key === "rewatch_engagement") {
      return (
        (Number(b.rewatch_engagement) || 0) -
        (Number(a.rewatch_engagement) || 0)
      );
    }
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

export function HistoryPage() {
  const [media, setMedia] = useState<MediaFilter>("all");
  const [sort, setSort] = useState<SortKey>("latest");
  const [search, setSearch] = useState("");
  const [posters, setPosters] = useState(true);
  const [rows, setRows] = useState<HistoryRow[] | null>(null);
  /** Unfiltered history for stats when media filter is movie/tv (overall totals). */
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
        if (r2.ok) {
          setRowsOverall((await r2.json()) as HistoryRow[]);
        } else {
          setRowsOverall(null);
        }
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
    if (q) {
      r = r.filter((row) => rowTitle(row).toLowerCase().includes(q));
    }
    return r;
  }, [rows, sort, search]);

  const statsSource = useMemo(() => {
    if (!rows) return [];
    if (media === "all") return rows;
    return rowsOverall ?? rows;
  }, [rows, rowsOverall, media]);

  const headlineCounts = useMemo(() => countMoviesShows(statsSource), [statsSource]);

  const watchStats = useMemo(
    () => computeWatchTimeMinutes(statsSource),
    [statsSource],
  );

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
            setAutoSyncNote("Background Trakt sync triggered (every 5 min).");
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
    <div className="page page-wide">
      <h1 className="page-title">Watch History</h1>
      <p className="lede">
        Filter and search your library. With this page open, Trakt is refreshed
        about every five minutes. Use <strong>Sync Trakt Now</strong> for an
        immediate update.
      </p>

      {rows && rows.length > 0 && (
        <div className="history-stats card">
          <div className="history-stat-metric">
            <h3>Movies Watched</h3>
            <p className="history-stat-num">{headlineCounts.movies}</p>
            <p className="muted small">
              {fmtMinutes(watchStats.movie)} (incl. rewatches)
            </p>
          </div>
          <div className="history-stat-metric">
            <h3>Shows Watched</h3>
            <p className="history-stat-num">{headlineCounts.tv}</p>
            <p className="muted small">
              {fmtMinutes(watchStats.show)} (incl. rewatches)
            </p>
          </div>
          <div className="history-stat-metric">
            <h3>Total Watch Time</h3>
            <p className="history-stat-num">
              {(watchStats.total / (60 * 24)).toFixed(1)} d
            </p>
            <p className="muted small">
              {fmtMinutes(watchStats.total)} (incl. rewatches)
            </p>
          </div>
        </div>
      )}

      <div className="toolbar card">
        <div className="toolbar-row">
          <label className="field">
            <span className="field-label">Media</span>
            <select
              className="input"
              value={media}
              onChange={(e) => setMedia(e.target.value as MediaFilter)}
            >
              <option value="all">All</option>
              <option value="movie">Movies</option>
              <option value="tv">TV</option>
            </select>
          </label>
          <label className="field">
            <span className="field-label">Sort</span>
            <select
              className="input"
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
          <label className="field field-grow">
            <span className="field-label">Search</span>
            <input
              className="input"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Title…"
            />
          </label>
          <label className="field checkbox-field">
            <input
              type="checkbox"
              checked={posters}
              onChange={(e) => setPosters(e.target.checked)}
            />
            <span>Poster Cards</span>
          </label>
        </div>
        <div className="toolbar-row">
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => void syncNow(true)}
          >
            {busy ? "Syncing…" : "Sync Trakt Now"}
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => void load()}
          >
            Reload
          </button>
          <span className="muted toolbar-stats">
                       Showing {listCount} of {totalRows} in view · Catalog: Movies{" "}
            {headlineCounts.movies} · TV {headlineCounts.tv}
          </span>
        </div>
        {syncNote && <p className="hint">{syncNote}</p>}
        {autoSyncNote && <p className="hint">{autoSyncNote}</p>}
      </div>

      {err && <div className="card card-error">{err}</div>}

      {rows === null && !err && <p className="muted">Loading…</p>}

      {rows && rows.length === 0 && (
        <p className="muted">No history yet. Sign in and sync Trakt.</p>
      )}

      {filtered.length > 0 && posters && (
        <div className="history-card-list">
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
              <article key={`${id}-${mt}-${i}`} className="history-card">
                <div className="history-card-poster">
                  <img src={src} alt="" loading="lazy" />
                </div>
                <div className="history-card-main">
                  <span className={`history-card-kind ${isTv ? "tv" : "movie"}`}>
                    {isTv ? "TV" : "Film"}
                  </span>
                  <h3 className="history-card-title">{title}</h3>
                  {isTv ? (
                    <p className="history-card-sub muted">
                      {String(row.watched_episodes ?? 0)} /{" "}
                      {String(row.total_episodes ?? "—")} episodes
                    </p>
                  ) : (
                    <p className="history-card-sub muted">
                      {row.year != null ? String(row.year) : "Year —"}
                    </p>
                  )}
                </div>
                <ul className="history-card-statline">
                  {isTv ? (
                    <li>
                      <span className="stat-label">Complete</span>
                      <span className="stat-value">
                        {`${((Number(row.completion_ratio) || 0) * 100).toFixed(0)}%`}
                      </span>
                    </li>
                  ) : (
                    <li>
                      <span className="stat-label">Watches</span>
                      <span className="stat-value">
                        {String(row.watch_count ?? 0)}
                      </span>
                    </li>
                  )}
                  {re > 0 && (
                    <li>
                      <span className="stat-label">Rewatch</span>
                      <span className="stat-value">×{re.toFixed(1)}</span>
                    </li>
                  )}
                  <li>
                    <span className="stat-label">Last</span>
                    <span className="stat-value mono">
                      {String(row.latest_watched_at ?? "—").slice(0, 10)}
                    </span>
                  </li>
                </ul>
                <div className="history-card-actions">
                  {id > 0 && mt && (
                    <Link
                      className="btn btn-secondary history-card-link"
                      to={`/similar?id=${id}&type=${mtEnc}`}
                    >
                      Similar Titles
                    </Link>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}

      {filtered.length > 0 && !posters && (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Title</th>
                <th>Type</th>
                <th>Latest</th>
                <th>Count</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => {
                const id = rowId(row);
                const mt = String(row.media_type ?? "");
                const mtEnc = encodeURIComponent(mt === "tv" ? "tv" : "movie");
                return (
                  <tr key={`${row.id}-${row.media_type}-${i}`}>
                    <td>{rowTitle(row)}</td>
                    <td>{mt || "—"}</td>
                    <td className="mono">
                      {String(row.latest_watched_at ?? "—")}
                    </td>
                    <td>{String(row.watch_count ?? "—")}</td>
                    <td>
                      {id > 0 && mt ? (
                        <Link
                          className="table-link"
                          to={`/similar?id=${id}&type=${mtEnc}`}
                        >
                          Similar Titles
                        </Link>
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
