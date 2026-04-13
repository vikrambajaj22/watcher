import type { HistoryRow } from "../api/watcher";

export function fmtMinutes(m: number): string {
  const mm = Math.floor(Math.max(0, m));
  const days = Math.floor(mm / (60 * 24));
  const hours = Math.floor((mm - days * 24 * 60) / 60);
  return `${mm.toLocaleString()} min (${days} days and ${hours} hrs)`;
}

/** Total watch minutes (movies: runtime * watch_count; TV: episode runtime * episode watch estimates). */
export function computeWatchTimeMinutes(items: HistoryRow[]): {
  total: number;
  movie: number;
  show: number;
} {
  let total = 0;
  let movie = 0;
  let show = 0;
  for (const it of items) {
    try {
      if (it.media_type === "movie") {
        const runtime = Number(it.runtime_minutes) || 0;
        const watchCount = Number(it.watch_count) || 0;
        const mins = runtime * watchCount;
        total += mins;
        movie += mins;
      } else {
        const epRuntime = Number(it.episode_runtime_minutes) || 0;
        let epWatchCount = Number(it.episode_watch_count) || 0;
        if (epWatchCount === 0) {
          const watchedEps = Number(it.watched_episodes) || 0;
          const reEng = Number(it.rewatch_engagement) || 0;
          const completion = Number(it.completion_ratio) || 0;
          let avgWatches = 1;
          if (completion > 0 && reEng > 0) avgWatches = reEng / completion;
          epWatchCount = Math.floor(watchedEps * avgWatches);
        }
        const mins = epRuntime * epWatchCount;
        total += mins;
        show += mins;
      }
    } catch {
      /* skip row */
    }
  }
  return { total, movie, show };
}

export function countMoviesShows(items: HistoryRow[]): {
  movies: number;
  tv: number;
} {
  return {
    movies: items.filter((x) => x.media_type === "movie").length,
    tv: items.filter((x) => x.media_type === "tv").length,
  };
}
