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

type Tab = "status" | "sync" | "cache";

type SyncJobRow = {
  key: string;
  job_type?: string;
  status?: string;
  processed?: number;
  started_at?: number;
  last_update?: number;
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
}: {
  health: Health | null;
  syncStatus: SyncStatus | null;
}) {
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
        <h2>Last Sync</h2>
        {syncStatus ? (
          <ul className="sync-list admin-sync-list">
            <li>
              <span className="muted">Trakt</span>{" "}
              {formatDisplayTs(syncStatus.trakt_last_activity ?? null)}
            </li>
          </ul>
        ) : (
          <p className="muted">Could not load sync status.</p>
        )}
        <RawJsonDetails label="Raw JSON" data={syncStatus} />
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

  const [jobs, setJobs] = useState<SyncJobRow[]>([]);
  const [selectedJob, setSelectedJob] = useState("");
  const [jobProbe, setJobProbe] = useState<JobStatus | null>(null);

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
        "Couldn't reach maintenance endpoints. If the server restricts these actions, configure access using the repository developer docs.",
      );
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

  async function pollSelected() {
    if (!selectedJob) {
      flash("Select a job key first", true);
      return;
    }
    const raw = selectedJob.includes(":") ? selectedJob.split(":").pop() : selectedJob;
    if (!raw) return;
    const st = await pollJobUntil(raw, "trakt", { maxWaitSec: 0 });
    setJobProbe(st);
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
        Sync jobs and cache controls for people operating this instance.
      </p>

      <div className="tabs">
        {(
          [
            ["status", "Status"],
            ["sync", "Sync Jobs"],
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
        <StatusTab health={health} syncStatus={syncStatus} />
      )}

      {tab === "sync" && (
        <div className="card">
          <h2>Trakt</h2>
          <div className="actions">
            <button type="button" className="btn btn-primary" onClick={() => void startTrakt()}>
              Start Trakt Sync
            </button>
          </div>

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
              onClick={() => void pollSelected()}
            >
              Probe Job
            </button>
          </div>
          {jobProbe && (
            <pre className="json-pre">{JSON.stringify(jobProbe, null, 2)}</pre>
          )}
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
