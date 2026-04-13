import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "../api/client";
import {
  apiJson,
  pollJobUntil,
  type Health,
  type JobStatus,
  type SyncStatus,
} from "../api/watcher";
import { formatDisplayTs } from "../lib/formatTs";

type Tab = "status" | "sync" | "faiss" | "cache";

type SyncJobRow = {
  key: string;
  job_type?: string;
  status?: string;
  processed?: number;
  embed_queued?: number;
  started_at?: number;
  last_update?: number;
};

type FaissStatusShape = {
  present?: boolean;
  cached?: boolean;
  source?: string;
  mount_path?: string;
  sidecar_meta?: Record<string, unknown> | null;
  paths?: Record<string, string>;
};

function RawJsonDetails({ label, data }: { label: string; data: unknown }) {
  if (data === null || data === undefined) return null;
  return (
    <details className="admin-raw-details">
      <summary>{label}</summary>
      <pre className="json-pre admin-raw-pre">
        {JSON.stringify(data, null, 2)}
      </pre>
    </details>
  );
}

function StatusTab({
  health,
  syncStatus,
  faissStatus,
}: {
  health: Health | null;
  syncStatus: SyncStatus | null;
  faissStatus: unknown;
}) {
  const f = faissStatus as FaissStatusShape | null;
  const meta = f?.sidecar_meta;

  return (
    <div className="grid-2 admin-status-grid">
      <div className="card admin-status-card">
        <h2>Server Health</h2>
        {health ? (
          <p className="admin-status-lead">
            <span className="badge badge-ok">{health.status}</span>
            {health.service && (
              <span className="muted"> · {health.service}</span>
            )}
          </p>
        ) : (
          <p className="muted">No health response yet.</p>
        )}
        <RawJsonDetails label="Raw JSON" data={health} />
      </div>

      <div className="card admin-status-card">
        <h2>Last Sync Times</h2>
        {syncStatus ? (
          <ul className="sync-list admin-sync-list">
            <li>
              <span className="muted">Trakt</span>{" "}
              {formatDisplayTs(syncStatus.trakt_last_activity ?? null)}
            </li>
            <li>
              <span className="muted">TMDB Movies</span>{" "}
              {formatDisplayTs(syncStatus.tmdb_movie_last_sync ?? null)}
            </li>
            <li>
              <span className="muted">TMDB TV</span>{" "}
              {formatDisplayTs(syncStatus.tmdb_tv_last_sync ?? null)}
            </li>
          </ul>
        ) : (
          <p className="muted">Could not load sync status.</p>
        )}
        <RawJsonDetails label="Raw JSON" data={syncStatus} />
      </div>

      <div className="card admin-status-card admin-status-wide">
        <h2>Search Index</h2>
        {f ? (
          <ul className="admin-kv-list">
            <li>
              <span className="admin-kv-label">Index On Disk</span>
              <span className="admin-kv-value">
                {f.present ? (
                  <span className="badge badge-ok">Present</span>
                ) : (
                  <span className="badge badge-warn">Missing</span>
                )}
              </span>
            </li>
            <li>
              <span className="admin-kv-label">Loaded In This Process</span>
              <span className="admin-kv-value">
                {f.cached ? (
                  <span className="badge badge-ok">Yes</span>
                ) : (
                  <span className="muted">No</span>
                )}
              </span>
            </li>
            {f.source && (
              <li>
                <span className="admin-kv-label">Source</span>
                <span className="admin-kv-value mono">{f.source}</span>
              </li>
            )}
            {f.mount_path && (
              <li>
                <span className="admin-kv-label">Mount Path</span>
                <span className="admin-kv-value mono">{f.mount_path}</span>
              </li>
            )}
            {meta && typeof meta === "object" && (
              <>
                {(meta.embedding_model != null || meta.model_name != null) && (
                  <li>
                    <span className="admin-kv-label">Embedding Model</span>
                    <span className="admin-kv-value">
                      {String(meta.embedding_model ?? meta.model_name)}
                    </span>
                  </li>
                )}
                {meta.num_vectors != null && (
                  <li>
                    <span className="admin-kv-label">Vectors</span>
                    <span className="admin-kv-value">
                      {String(meta.num_vectors)}
                    </span>
                  </li>
                )}
                {(meta.embedding_dims != null || meta.dims != null) && (
                  <li>
                    <span className="admin-kv-label">Dimensions</span>
                    <span className="admin-kv-value">
                      {String(meta.embedding_dims ?? meta.dims)}
                    </span>
                  </li>
                )}
              </>
            )}
          </ul>
        ) : (
          <p className="muted">Could not load index status.</p>
        )}
        <RawJsonDetails label="Raw JSON" data={faissStatus} />
      </div>
    </div>
  );
}

export function AdminPage() {
  const [tab, setTab] = useState<Tab>("status");
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [health, setHealth] = useState<Health | null>(null);
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  const [faissStatus, setFaissStatus] = useState<unknown>(null);

  const [jobs, setJobs] = useState<SyncJobRow[]>([]);
  const [selectedJob, setSelectedJob] = useState("");
  const [jobProbe, setJobProbe] = useState<JobStatus | null>(null);

  const [tmdbMedia, setTmdbMedia] = useState<"movie" | "tv">("movie");
  const [tmdbFull, setTmdbFull] = useState(false);
  const [tmdbEmbed, setTmdbEmbed] = useState(true);
  const [tmdbForce, setTmdbForce] = useState(false);
  const [lastTmdbJob, setLastTmdbJob] = useState("");

  const [embedId, setEmbedId] = useState(550);
  const [embedMedia, setEmbedMedia] = useState<"movie" | "tv">("movie");
  const [embedForce, setEmbedForce] = useState(false);

  const [faissDim, setFaissDim] = useState(384);
  const [faissFactory, setFaissFactory] = useState("IDMap,IVF100,Flat");
  const [faissForce, setFaissForce] = useState(false);

  const refreshMeta = useCallback(async () => {
    setErr(null);
    try {
      const rh = await apiFetch("/health");
      if (rh.ok) {
        setHealth((await rh.json()) as Health);
      } else {
        setHealth(null);
      }
    } catch {
      setHealth(null);
    }
    try {
      setSyncStatus(await apiJson<SyncStatus>("/admin/sync/status"));
    } catch {
      setSyncStatus(null);
      setErr(
        "Couldn’t reach maintenance endpoints. If the server restricts these actions, configure access using the repository developer docs.",
      );
    }
    try {
      setFaissStatus(await apiJson("/admin/faiss/status"));
    } catch {
      setFaissStatus(null);
    }
    try {
      const j = await apiJson<{ jobs: SyncJobRow[] }>("/admin/sync/jobs");
      setJobs(j.jobs ?? []);
    } catch {
      setJobs([]);
    }
  }, []);

  useEffect(() => {
    void refreshMeta();
  }, [refreshMeta]);

  function flash(m: string, isErr = false) {
    if (isErr) setErr(m);
    else {
      setErr(null);
      setMsg(m);
    }
  }

  async function postJson(path: string, body: unknown) {
    const r = await apiFetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    const text = await r.text();
    if (!r.ok) throw new Error(text);
    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  }

  async function startTrakt() {
    setMsg(null);
    try {
      const j = (await postJson("/admin/sync/trakt", {})) as { job_id?: string };
      flash(`Trakt job: ${j.job_id ?? "accepted"}`);
      await refreshMeta();
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  async function pollSelected(jobType: "trakt" | "tmdb") {
    if (!selectedJob) {
      flash("Select A Job Key First", true);
      return;
    }
    const raw = selectedJob.includes(":") ? selectedJob.split(":").pop() : selectedJob;
    if (!raw) return;
    const st = await pollJobUntil(raw, jobType, { maxWaitSec: 0 });
    setJobProbe(st);
  }

  async function startTmdb() {
    setMsg(null);
    try {
      const j = (await postJson("/admin/sync/tmdb", {
        media_type: tmdbMedia,
        full_sync: tmdbFull,
        embed_updated: tmdbEmbed,
        force_refresh: tmdbForce,
      })) as { job_id?: string };
      if (j.job_id) setLastTmdbJob(j.job_id);
      flash(`TMDB job: ${j.job_id ?? "accepted"}`);
      await refreshMeta();
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  async function cancelTmdb(jobId?: string) {
    const jid = jobId || lastTmdbJob;
    if (!jid) {
      flash("No Job ID", true);
      return;
    }
    try {
      await postJson("/admin/sync/tmdb/cancel", { job_id: jid });
      flash("Cancel requested");
      await refreshMeta();
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  async function faissUpsert() {
    try {
      const r = await postJson("/admin/faiss/upsert-item", {
        id: embedId,
        media_type: embedMedia,
        force_regenerate: embedForce,
      });
      flash(JSON.stringify(r));
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  async function faissRebuild() {
    try {
      const r = await postJson("/admin/faiss/rebuild", {
        dim: faissDim,
        factory: faissFactory,
        force_regenerate: faissForce,
      });
      flash(JSON.stringify(r));
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  async function embedFull() {
    try {
      const r = await postJson("/admin/embed/full", {
        force_regenerate: faissForce,
      });
      flash(JSON.stringify(r));
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  async function embedItem() {
    try {
      const r = await postJson("/admin/embed/item", {
        id: embedId,
        media_type: embedMedia,
        force_regenerate: embedForce,
      });
      flash(JSON.stringify(r));
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  async function clearFaissCache() {
    try {
      await postJson("/admin/faiss/clear-cache", {});
      flash("Index Cache Cleared");
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  async function clearHistoryCache() {
    try {
      await postJson("/admin/clear-history-cache", {});
      flash("History Cache Cleared");
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  return (
    <div className="page page-wide">
      <h1 className="page-title">Maintenance</h1>
      <p className="lede">
        Sync jobs, search index, and cache controls for people operating this
        instance. Technical responses from the server are available under each
        section’s <strong>Raw JSON</strong>.
      </p>

      <div className="tabs">
        {(
          [
            ["status", "Status"],
            ["sync", "Sync Jobs"],
            ["faiss", "Embeddings & Index"],
            ["cache", "Cache"],
          ] as const
        ).map(([k, label]) => (
          <button
            key={k}
            type="button"
            className={tab === k ? "tab active" : "tab"}
            onClick={() => setTab(k)}
          >
            {label}
          </button>
        ))}
      </div>

      {msg && <div className="card admin-msg">{msg}</div>}
      {err && <div className="card card-error">{err}</div>}

      {tab === "status" && (
        <StatusTab
          health={health}
          syncStatus={syncStatus}
          faissStatus={faissStatus}
        />
      )}

      {tab === "sync" && (
        <div className="card">
          <h2>Trakt</h2>
          <div className="actions">
            <button type="button" className="btn btn-primary" onClick={() => void startTrakt()}>
              Start Trakt Sync
            </button>
          </div>
          <h2>TMDB</h2>
          <div className="toolbar-row">
            <label className="field">
              <span className="field-label">Media</span>
              <select
                className="input"
                value={tmdbMedia}
                onChange={(e) => setTmdbMedia(e.target.value as "movie" | "tv")}
              >
                <option value="movie">Movie</option>
                <option value="tv">TV</option>
              </select>
            </label>
            <label className="field checkbox-field">
              <input
                type="checkbox"
                checked={tmdbFull}
                onChange={(e) => setTmdbFull(e.target.checked)}
              />
              <span>Full Sync</span>
            </label>
            <label className="field checkbox-field">
              <input
                type="checkbox"
                checked={tmdbEmbed}
                onChange={(e) => setTmdbEmbed(e.target.checked)}
              />
              <span>Embed Updated</span>
            </label>
            <label className="field checkbox-field">
              <input
                type="checkbox"
                checked={tmdbForce}
                onChange={(e) => setTmdbForce(e.target.checked)}
              />
              <span>Force Refresh</span>
            </label>
          </div>
          <div className="actions">
            <button type="button" className="btn btn-primary" onClick={() => void startTmdb()}>
              Start TMDB Sync
            </button>
            <button type="button" className="btn btn-ghost" onClick={() => void cancelTmdb()}>
              Cancel Last TMDB Job
            </button>
          </div>
          {lastTmdbJob && (
            <p className="hint">Last TMDB Job ID: {lastTmdbJob}</p>
          )}

          <h2>Active / Recent Jobs</h2>
          <button type="button" className="btn btn-ghost" onClick={() => void refreshMeta()}>
            Refresh List
          </button>
          <div className="table-wrap" style={{ marginTop: "0.75rem" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Processed</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.key}>
                    <td className="mono">{j.key}</td>
                    <td>{j.job_type}</td>
                    <td>{j.status}</td>
                    <td>{j.processed ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {jobs.length === 0 && <p className="muted">No Pending Jobs.</p>}
          </div>

          <h2>Poll Job</h2>
          <label className="field field-block">
            <span className="field-label">Job ID Or Full Key</span>
            <input
              className="input mono"
              value={selectedJob}
              onChange={(e) => setSelectedJob(e.target.value)}
              placeholder="uuid or trakt_sync_job:…"
            />
          </label>
          <div className="actions">
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => void pollSelected("trakt")}
            >
              Probe Trakt Job
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => void pollSelected("tmdb")}
            >
              Probe TMDB Job
            </button>
          </div>
          {jobProbe && (
            <pre className="json-pre">{JSON.stringify(jobProbe, null, 2)}</pre>
          )}
        </div>
      )}

      {tab === "faiss" && (
        <div className="card">
          <h2>Single Item</h2>
          <div className="toolbar-row">
            <label className="field">
              <span className="field-label">TMDB ID</span>
              <input
                className="input input-narrow"
                type="number"
                value={embedId}
                onChange={(e) => setEmbedId(Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span className="field-label">Type</span>
              <select
                className="input"
                value={embedMedia}
                onChange={(e) =>
                  setEmbedMedia(e.target.value as "movie" | "tv")
                }
              >
                <option value="movie">Movie</option>
                <option value="tv">TV</option>
              </select>
            </label>
            <label className="field checkbox-field">
              <input
                type="checkbox"
                checked={embedForce}
                onChange={(e) => setEmbedForce(e.target.checked)}
              />
              <span>Force Regenerate</span>
            </label>
          </div>
          <div className="actions">
            <button type="button" className="btn btn-primary" onClick={() => void faissUpsert()}>
              Incremental Upsert
            </button>
            <button type="button" className="btn btn-ghost" onClick={() => void embedItem()}>
              Embed Item (Full Pipeline)
            </button>
          </div>

          <h2>Full Rebuild</h2>
          <div className="toolbar-row">
            <label className="field">
              <span className="field-label">Dim</span>
              <input
                className="input input-narrow"
                type="number"
                value={faissDim}
                onChange={(e) => setFaissDim(Number(e.target.value))}
              />
            </label>
            <label className="field field-grow">
              <span className="field-label">Factory</span>
              <input
                className="input mono"
                value={faissFactory}
                onChange={(e) => setFaissFactory(e.target.value)}
              />
            </label>
            <label className="field checkbox-field">
              <input
                type="checkbox"
                checked={faissForce}
                onChange={(e) => setFaissForce(e.target.checked)}
              />
              <span>Force Regenerate</span>
            </label>
          </div>
          <div className="actions">
            <button type="button" className="btn btn-primary" onClick={() => void faissRebuild()}>
              Rebuild Index (Background Subprocess)
            </button>
            <button type="button" className="btn btn-ghost" onClick={() => void embedFull()}>
              Full Embeddings Rebuild (In-Process Task)
            </button>
            <button type="button" className="btn btn-ghost" onClick={() => void clearFaissCache()}>
              Clear In-Memory Index Cache
            </button>
          </div>
        </div>
      )}

      {tab === "cache" && (
        <div className="card">
          <p>Clear server-side history cache after a manual sync.</p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => void clearHistoryCache()}
          >
            Clear History Cache
          </button>
        </div>
      )}
    </div>
  );
}
