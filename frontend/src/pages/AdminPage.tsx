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

const inputCls = "glass-input rounded-lg text-text px-2.5 py-2 text-sm w-full";

function RawJsonDetails({ label, data }: { label: string; data: unknown }) {
  if (data === null || data === undefined) return null;
  return (
    <details className="mt-3">
      <summary className="cursor-pointer text-sm text-accent select-none">{label}</summary>
      <pre className="mt-2 font-mono text-[0.78rem] overflow-auto max-h-[200px] whitespace-pre-wrap break-words">
        {JSON.stringify(data, null, 2)}
      </pre>
    </details>
  );
}

function StatusTab({ health, syncStatus }: { health: Health | null; syncStatus: SyncStatus | null }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div className="p-5 glass rounded-2xl">
        <h2 className="text-[0.85rem] font-semibold uppercase tracking-[0.06em] text-muted mb-3">
          Server Health
        </h2>
        {health ? (
          <p className="m-0">
            <span className="px-2 py-0.5 rounded text-sm font-semibold bg-emerald-400/15 text-emerald-300">
              {health.status}
            </span>
            {health.service && <span className="text-muted"> · {health.service}</span>}
          </p>
        ) : (
          <p className="text-muted m-0">No health response yet.</p>
        )}
        <RawJsonDetails label="Raw JSON" data={health} />
      </div>

      <div className="p-5 glass rounded-2xl">
        <h2 className="text-[0.85rem] font-semibold uppercase tracking-[0.06em] text-muted mb-3">
          Last Sync
        </h2>
        {syncStatus ? (
          <ul className="list-none m-0 p-0 flex flex-col gap-1.5 text-sm">
            <li>
              <span className="text-muted">Trakt </span>
              {formatDisplayTs(syncStatus.trakt_last_activity ?? null)}
            </li>
          </ul>
        ) : (
          <p className="text-muted m-0">Could not load sync status.</p>
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
      if (rh.ok) setHealth((await rh.json()) as Health);
      else setHealth(null);
    } catch {
      setHealth(null);
    }
    try {
      setSyncStatus(await apiJson<SyncStatus>("/admin/sync/status"));
    } catch {
      setSyncStatus(null);
      setErr(
        "Couldn't reach maintenance endpoints. Configure access using the developer docs.",
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
    else { setErr(null); setMsg(m); }
  }

  async function postJson(path: string, body: unknown) {
    const r = await apiFetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    const text = await r.text();
    if (!r.ok) throw new Error(text);
    try { return JSON.parse(text); } catch { return { raw: text }; }
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
    if (!selectedJob) { flash("Select a job key first", true); return; }
    const raw = selectedJob.includes(":") ? selectedJob.split(":").pop() : selectedJob;
    if (!raw) return;
    const st = await pollJobUntil(raw, "trakt", { maxWaitSec: 0 });
    setJobProbe(st);
  }

  async function clearHistoryCache() {
    try {
      await postJson("/admin/clear-history-cache", {});
      flash("History cache cleared.");
    } catch (e) {
      flash(e instanceof Error ? e.message : "Failed", true);
    }
  }

  const tabs: { key: Tab; label: string }[] = [
    { key: "status", label: "Status" },
    { key: "sync", label: "Sync Jobs" },
    { key: "cache", label: "Cache" },
  ];

  return (
    <div className="w-full">
      <h1 className="text-[1.75rem] font-bold tracking-[-0.03em] mb-1.5">Maintenance</h1>
      <p className="text-muted mb-6">
        Sync jobs and cache controls for people operating this instance.
      </p>

      <div className="flex gap-1.5 mb-5 overflow-x-auto pb-1">
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            className={`px-3 py-2 rounded-lg text-sm font-medium cursor-pointer border transition-all shrink-0 ${
              tab === key
                ? "bg-accent/12 border-accent/40 text-text"
                : "bg-white/5 border-white/10 text-muted hover:text-text hover:border-white/20"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {msg && (
        <div className="p-4 glass rounded-2xl mb-4 font-mono text-sm">
          {msg}
        </div>
      )}
      {err && (
        <div className="p-4 glass border-danger/40 rounded-xl mb-4">
          <strong className="text-danger">Error: </strong>
          {err}
        </div>
      )}

      {tab === "status" && <StatusTab health={health} syncStatus={syncStatus} />}

      {tab === "sync" && (
        <div className="p-5 glass-dark rounded-2xl flex flex-col gap-5">
          <div>
            <h2 className="text-[0.85rem] font-semibold uppercase tracking-[0.06em] text-muted mb-3">
              Trakt
            </h2>
            <button
              type="button"
              className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-bg font-semibold text-sm cursor-pointer transition-all shadow-[inset_0_1px_0_rgba(255,255,255,0.3)] hover:brightness-110 hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.35),0_0_24px_-4px_rgba(74,222,128,0.45)] border-0"
              onClick={() => void startTrakt()}
            >
              Start Trakt Sync
            </button>
          </div>

          <div>
            <h2 className="text-[0.85rem] font-semibold uppercase tracking-[0.06em] text-muted mb-3">
              Active / Recent Jobs
            </h2>
            <button
              type="button"
              className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-transparent text-muted border border-border font-semibold text-sm cursor-pointer transition-colors hover:text-text hover:border-muted mb-3"
              onClick={() => void refreshMeta()}
            >
              Refresh List
            </button>
            <div className="overflow-x-auto border border-border rounded-xl">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr>
                    {["Key", "Type", "Status", "Processed"].map((h) => (
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
                  {jobs.map((j) => (
                    <tr key={j.key} className="border-b border-border last:border-b-0 hover:bg-accent/5 transition-colors">
                      <td className="px-4 py-3 font-mono text-[0.85em]">{j.key}</td>
                      <td className="px-4 py-3">{j.job_type}</td>
                      <td className="px-4 py-3">{j.status}</td>
                      <td className="px-4 py-3">{j.processed ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {jobs.length === 0 && (
                <p className="text-muted px-4 py-3 text-sm m-0">No pending jobs.</p>
              )}
            </div>
          </div>

          <div>
            <h2 className="text-[0.85rem] font-semibold uppercase tracking-[0.06em] text-muted mb-3">
              Poll Job
            </h2>
            <label className="flex flex-col gap-1.5 mb-3">
              <span className="text-[0.72rem] font-semibold uppercase tracking-[0.05em] text-muted">
                Job ID or Full Key
              </span>
              <input
                className={`${inputCls} font-mono`}
                value={selectedJob}
                onChange={(e) => setSelectedJob(e.target.value)}
                placeholder="uuid or trakt_sync_job:…"
              />
            </label>
            <button
              type="button"
              className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-transparent text-muted border border-border font-semibold text-sm cursor-pointer transition-colors hover:text-text hover:border-muted"
              onClick={() => void pollSelected()}
            >
              Probe Job
            </button>
            {jobProbe && (
              <pre className="mt-3 font-mono text-[0.78rem] overflow-auto max-h-[320px] whitespace-pre-wrap break-words">
                {JSON.stringify(jobProbe, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {tab === "cache" && (
        <div className="p-5 glass-dark rounded-2xl">
          <p className="text-sm text-muted mb-4">
            Clear server-side history cache after a manual sync.
          </p>
          <button
            type="button"
            className="inline-flex items-center justify-center px-4 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-bg font-semibold text-sm cursor-pointer transition-all shadow-[inset_0_1px_0_rgba(255,255,255,0.3)] hover:brightness-110 hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.35),0_0_24px_-4px_rgba(74,222,128,0.45)] border-0"
            onClick={() => void clearHistoryCache()}
          >
            Clear History Cache
          </button>
        </div>
      )}
    </div>
  );
}
