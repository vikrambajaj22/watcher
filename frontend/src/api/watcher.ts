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

export type TmdbRecommendResponse = {
  recommendations: Recommendation[];
  debug?: Record<string, unknown> | null;
};

export type SearchHit = {
  id: number;
  title: string;
  media_type: "movie" | "tv";
  year?: string | null;
  poster_path?: string | null;
  watched?: boolean;
};

export type PersonSuggestion = {
  id: number;
  name?: string | null;
  profile_path?: string | null;
  known_for_department?: string | null;
  known_for?: string[];
};

export type SimilarResult = {
  id: number;
  title?: string;
  media_type?: string;
  poster_path?: string;
  overview?: string;
  release_date?: string;
};

export type SimilarResponse = { source_title?: string | null; results: SimilarResult[] };

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

export type TasteProfile = {
  signature: string;
  summary: string;
  genres: string[];
  themes: string[];
  avoid: string[];
  history_count: number;
};

export type DiscoverItem = {
  id: number;
  title?: string;
  media_type?: string;
  poster_path?: string | null;
  overview?: string | null;
  release_date?: string | null;
  watched?: boolean;
};

export type DescribeFilters = {
  media_type?: string;
  genres?: string[];
  cast?: string[];
  keywords?: string[];
  year_from?: number;
  year_to?: number;
};

export type DescribeResponse = {
  results: DiscoverItem[];
  filters?: DescribeFilters | null;
};

export type PersonSummary = {
  id: number;
  name?: string;
  profile_path?: string | null;
  known_for_department?: string | null;
};

export type ActorHistoryItem = {
  id: number;
  title?: string;
  media_type?: string;
  poster_path?: string | null;
  character?: string | null;
  department?: string | null;
  watched_at?: string | null;
};

export type ActorHistoryResponse = {
  person: PersonSummary;
  items: ActorHistoryItem[];
};

export type ChatEventToken = { type: "message"; content: string };
export type ChatEventToolStart = { type: "tool_start"; tool: string; label: string; args?: Record<string, unknown>; run_id?: string };
export type ChatEventToolResult = { type: "tool_result"; tool: string; data: Record<string, unknown>; run_id?: string; duration_ms?: number };
export type ChatEventError = { type: "error"; message: string };
export type ChatEventDone = { type: "done" };
export type ChatEvent =
  | ChatEventToken
  | ChatEventToolStart
  | ChatEventToolResult
  | ChatEventError
  | ChatEventDone;

export async function* streamChat(threadId: string, message: string): AsyncGenerator<ChatEvent> {
  const r = await apiFetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, message }),
  });
  if (!r.body) return;
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6)) as ChatEvent;
        } catch {
          /* skip malformed */
        }
      }
    }
  }
}

export type WatchlistItem = {
  tmdb_id: number;
  media_type: string;
  title?: string | null;
  poster_path?: string | null;
  overview?: string | null;
  release_date?: string | null;
  genres?: string[] | null;
  synced_at?: string | null;
};

export type WatchlistSyncResult = { added: number; removed: number };

export function getHistoryQuery(mediaType: string | null, includePosters: boolean) {
  const p = new URLSearchParams();
  if (mediaType) p.set("media_type", mediaType);
  p.set("include_posters", includePosters ? "true" : "false");
  return `/history?${p.toString()}`;
}

export async function pollJobUntil(
  jobId: string,
  jobType: "trakt",
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
