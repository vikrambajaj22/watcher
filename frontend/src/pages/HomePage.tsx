import { useEffect } from "react";
import { Link } from "react-router-dom";
import { getApiBase } from "../api/client";
import { useAuth } from "../contexts/AuthContext";
import { useWatchlist } from "../contexts/WatchlistContext";

const FEATURES = [
  { label: "Watch History", desc: "Browse and filter your library by genre, year, and more" },
  { label: "Recommendations", desc: "AI-personalized picks based on your taste profile" },
  { label: "Discover", desc: "Find titles by describing what you want to watch" },
  { label: "Will I Like It?", desc: "Check any title against your watch history" },
  { label: "Similar Titles", desc: "Find movies and shows similar to anything you've seen" },
  { label: "Actor Search", desc: "See everything you've watched featuring a specific actor" },
  { label: "Chat", desc: "Ask Watcher anything in natural language" },
  { label: "Taste Profile", desc: "An AI-generated summary of your viewing taste" },
] as const;

export function HomePage() {
  const { authenticated, refresh: refreshAuth } = useAuth();
  const { watchlist } = useWatchlist();
  const loginHref = `${getApiBase()}/auth/trakt/start?from_ui=true`;
  const movieCount = watchlist.filter((w) => w.media_type === "movie").length;
  const tvCount = watchlist.filter((w) => w.media_type === "tv").length;

  useEffect(() => {
    void refreshAuth();
  }, [refreshAuth]);

  if (authenticated === null) {
    return <p className="text-muted">Loading…</p>;
  }

  return (
    <div className="relative flex flex-col items-center justify-center min-h-[75vh] text-center px-4 overflow-hidden">
      {/* Ambient background blobs */}
      <div className="absolute top-[-20%] left-1/2 -translate-x-1/2 w-[700px] h-[500px] rounded-full bg-accent/8 blur-[140px] pointer-events-none" />
      <div className="absolute top-[35%] left-[10%] w-[320px] h-[320px] rounded-full bg-blue-500/5 blur-[100px] pointer-events-none" />
      <div className="absolute top-[25%] right-[8%] w-[260px] h-[260px] rounded-full bg-accent/5 blur-[90px] pointer-events-none" />

      {/* Label pill */}
      <div className="mb-7 inline-flex items-center gap-1.5 px-3 py-1 rounded-full glass text-accent text-[0.7rem] font-bold uppercase tracking-widest">
        Trakt · TMDB · AI
      </div>

      {/* Heading */}
      <h1 className="text-[4rem] md:text-[5.5rem] font-bold tracking-[-0.05em] leading-[1.05] mb-5 bg-gradient-to-b from-white to-text/50 bg-clip-text text-transparent">
        Watcher
      </h1>

      {/* Tagline */}
      <p className="text-lg md:text-xl text-muted max-w-[44ch] leading-relaxed mb-8">
        {authenticated
          ? "AI recommendations, natural language discovery, taste analysis, similarity search, actor lookups, and more — all from your Trakt history."
          : "Your complete movie and TV companion. Sign in to unlock AI-powered discovery, recommendations, and more."}
      </p>

      {/* Watchlist widget */}
      {authenticated && watchlist.length > 0 && (
        <Link
          to="/watchlist"
          className="mb-8 inline-flex items-center gap-2.5 px-4 py-2.5 rounded-xl glass hover:bg-white/8 transition-all group"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent">
            <path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>
          </svg>
          <span className="text-sm font-medium text-text group-hover:text-accent transition-colors">
            {movieCount > 0 && tvCount > 0
              ? `${movieCount} movie${movieCount !== 1 ? "s" : ""} · ${tvCount} show${tvCount !== 1 ? "s" : ""} to watch`
              : movieCount > 0
              ? `${movieCount} movie${movieCount !== 1 ? "s" : ""} to watch`
              : `${tvCount} show${tvCount !== 1 ? "s" : ""} to watch`}
          </span>
          <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted group-hover:text-accent transition-colors">
            <path d="M9 18l6-6-6-6"/>
          </svg>
        </Link>
      )}

      {!authenticated && (
        <>
          <a
            className="inline-flex items-center gap-2 px-7 min-h-12 rounded-xl bg-gradient-to-br from-accent to-accent-dim text-bg font-semibold text-[0.95rem] shadow-[inset_0_1px_0_rgba(255,255,255,0.3)] hover:brightness-110 hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.35),0_0_30px_-4px_rgba(74,222,128,0.5)] transition-all mb-16"
            href={loginHref}
          >
            Sign in with Trakt →
          </a>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl w-full text-left">
            {FEATURES.map(({ label, desc }) => (
              <div key={label} className="glass rounded-2xl p-4">
                <p className="text-sm font-semibold text-text mb-1">{label}</p>
                <p className="text-xs text-muted leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
