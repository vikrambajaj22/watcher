import { apiFetch } from "./client";

export async function parseApiError(r: Response): Promise<string> {
  try {
    const j = (await r.json()) as { detail?: unknown };
    const d = j.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((x: { msg?: string }) => x?.msg ?? JSON.stringify(x))
        .join("; ");
    }
    if (d && typeof d === "object") return JSON.stringify(d);
  } catch {
    /* ignore */
  }
  return `${r.status} ${r.statusText}`;
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await apiFetch(path, init);
  if (!r.ok) throw new Error(await parseApiError(r));
  return r.json() as Promise<T>;
}

export async function apiJsonAllow<T>(
  path: string,
  init?: RequestInit,
): Promise<{ ok: true; data: T } | { ok: false; status: number; message: string }> {
  const r = await apiFetch(path, init);
  if (!r.ok) {
    return { ok: false, status: r.status, message: await parseApiError(r) };
  }
  return { ok: true, data: (await r.json()) as T };
}

export type Health = { status: string; service?: string };
export type AuthStatus = { authenticated: boolean };
export type SyncStatus = {
  trakt_last_activity?: string | null;
  tmdb_movie_last_sync?: string | null;
  tmdb_tv_last_sync?: string | null;
};
export type JobStatus = {
  status?: string;
  started_at?: number;
  finished_at?: number;
  processed?: number;
  embed_queued?: number;
  last_update?: number;
  result?: Record<string, unknown>;
  error?: string;
};

export type HistoryRow = Record<string, unknown>;

export type Recommendation = {
  id: string;
  title: string;
  reasoning: string;
  media_type?: string;
  metadata?: Record<string, unknown>;
};

export type RecommendResponse = { recommendations: Recommendation[] };

export type KnnResult = {
  id: number;
  title?: string;
  media_type?: string;
  score?: number;
  poster_path?: string;
  overview?: string;
};

export type KnnResponse = { results: KnnResult[] };

export type WillLikeResponse = {
  will_like: boolean;
  score: number;
  explanation: string;
  already_watched?: boolean;
  item: {
    id?: number;
    title?: string;
    name?: string;
    media_type?: string;
    overview?: string;
    poster_path?: string;
  };
};

export type ClusterItem = {
  id?: number;
  title?: string;
  media_type?: string;
  poster_path?: string;
  x: number;
  y: number;
  cluster: number;
  watch_count?: number;
  completion_ratio?: number;
  overview?: string;
  genres?: unknown[];
};

export type ClusterSummaries = Record<
  string,
  {
    size?: number;
    sample_titles?: string[];
    movie_count?: number;
    tv_count?: number;
    name?: string;
    top_genres?: string[];
  }
>;

export type ClustersResponse = {
  items: ClusterItem[];
  cluster_summaries: ClusterSummaries;
  total_items?: number;
  total_in_history?: number;
  n_clusters?: number;
  method?: string;
};

export function getHistoryQuery(mediaType: string | null, includePosters: boolean) {
  const p = new URLSearchParams();
  if (mediaType) p.set("media_type", mediaType);
  p.set("include_posters", includePosters ? "true" : "false");
  return `/history?${p.toString()}`;
}

export async function pollJobUntil(
  jobId: string,
  jobType: "trakt" | "tmdb",
  opts: { maxWaitSec?: number; intervalSec?: number } = {},
): Promise<JobStatus | null> {
  const maxWait = opts.maxWaitSec ?? 120;
  const interval = opts.intervalSec ?? 2;

  async function probe(): Promise<JobStatus | null> {
    const r = await apiFetch(
      `/admin/sync/job/${encodeURIComponent(jobId)}?job_type=${jobType}`,
    );
    if (!r.ok) return null;
    return (await r.json()) as JobStatus;
  }

  if (maxWait <= 0) {
    return probe();
  }

  let waited = 0;
  while (waited <= maxWait) {
    const j = await probe();
    if (j) {
      const s = j.status;
      if (s === "completed" || s === "failed" || s === "canceled") return j;
    }
    await new Promise((res) => setTimeout(res, interval * 1000));
    waited += interval;
  }
  return null;
}
