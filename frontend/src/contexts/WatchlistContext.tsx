import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { apiFetch } from "../api/client";
import { type WatchlistItem } from "../api/watcher";
import { useAuth } from "./AuthContext";

type ToggleTarget = {
  id: number;
  title: string;
  mediaType: string;
  posterPath?: string | null;
  overview?: string | null;
  releaseDate?: string | null;
};

type WatchlistContextValue = {
  watchlist: WatchlistItem[];
  isOnWatchlist: (id: number, mediaType: string) => boolean;
  toggle: (item: ToggleTarget) => Promise<void>;
  sync: () => Promise<{ added: number; removed: number }>;
  isToggling: (id: number, mediaType: string) => boolean;
};

const WatchlistContext = createContext<WatchlistContextValue | null>(null);

export function WatchlistProvider({ children }: { children: ReactNode }) {
  const { authenticated } = useAuth();
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [togglingIds, setTogglingIds] = useState<Set<string>>(new Set());

  const watchlistKey = (id: number, mt: string) => `${id}-${mt}`;

  const fetchWatchlist = useCallback(async () => {
    try {
      const r = await apiFetch("/watchlist");
      if (!r.ok) return;
      setWatchlist((await r.json()) as WatchlistItem[]);
    } catch {
      /* silently ignore */
    }
  }, []);

  useEffect(() => {
    if (authenticated) void fetchWatchlist();
    else setWatchlist([]);
  }, [authenticated, fetchWatchlist]);

  const isOnWatchlist = useCallback(
    (id: number, mediaType: string) =>
      watchlist.some((w) => w.tmdb_id === id && w.media_type === mediaType),
    [watchlist],
  );

  const isToggling = useCallback(
    (id: number, mediaType: string) => togglingIds.has(watchlistKey(id, mediaType)),
    [togglingIds],
  );

  const toggle = useCallback(
    async (item: ToggleTarget) => {
      const key = watchlistKey(item.id, item.mediaType);
      const onList = watchlist.some(
        (w) => w.tmdb_id === item.id && w.media_type === item.mediaType,
      );

      // Optimistic update
      if (onList) {
        setWatchlist((prev) =>
          prev.filter((w) => !(w.tmdb_id === item.id && w.media_type === item.mediaType)),
        );
      } else {
        const optimistic: WatchlistItem = {
          tmdb_id: item.id,
          media_type: item.mediaType,
          title: item.title,
          poster_path: item.posterPath,
          overview: item.overview,
          release_date: item.releaseDate,
        };
        setWatchlist((prev) => [optimistic, ...prev]);
      }

      setTogglingIds((prev) => new Set(prev).add(key));
      try {
        if (onList) {
          const r = await apiFetch(
            `/watchlist/${item.id}?media_type=${encodeURIComponent(item.mediaType)}`,
            { method: "DELETE" },
          );
          if (!r.ok) throw new Error();
        } else {
          const r = await apiFetch("/watchlist", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              tmdb_id: item.id,
              media_type: item.mediaType,
              title: item.title,
              poster_path: item.posterPath,
              overview: item.overview,
              release_date: item.releaseDate,
            }),
          });
          if (!r.ok) throw new Error();
          const saved = (await r.json()) as WatchlistItem;
          setWatchlist((prev) =>
            prev.map((w) =>
              w.tmdb_id === item.id && w.media_type === item.mediaType ? saved : w,
            ),
          );
        }
      } catch {
        // Revert on failure
        await fetchWatchlist();
      } finally {
        setTogglingIds((prev) => {
          const next = new Set(prev);
          next.delete(key);
          return next;
        });
      }
    },
    [watchlist, fetchWatchlist],
  );

  const sync = useCallback(async () => {
    const r = await apiFetch("/watchlist/sync", { method: "POST" });
    if (!r.ok) throw new Error("Sync failed");
    const result = (await r.json()) as { added: number; removed: number };
    await fetchWatchlist();
    return result;
  }, [fetchWatchlist]);

  return (
    <WatchlistContext.Provider value={{ watchlist, isOnWatchlist, toggle, sync, isToggling }}>
      {children}
    </WatchlistContext.Provider>
  );
}

export function useWatchlist(): WatchlistContextValue {
  const ctx = useContext(WatchlistContext);
  if (!ctx) throw new Error("useWatchlist must be used within WatchlistProvider");
  return ctx;
}
