import { useEffect } from "react";
import { Link } from "react-router-dom";
import { getApiBase } from "../api/client";
import { useAuth } from "../contexts/AuthContext";

const APP_SECTIONS = [
  { to: "/history", label: "Watch History", desc: "Browse and filter your full library" },
  { to: "/recommend", label: "Recommendations", desc: "AI picks based on your taste" },
  { to: "/will-like", label: "Will I Like It?", desc: "Check a title against your history" },
  { to: "/similar", label: "Similar Titles", desc: "Find movies and shows like one you love" },
  { to: "/admin", label: "Maintenance", desc: "Sync Trakt and manage cache" },
] as const;

export function HomePage() {
  const { authenticated, refresh: refreshAuth } = useAuth();
  const loginHref = `${getApiBase()}/auth/trakt/start?from_ui=true`;

  useEffect(() => {
    void refreshAuth();
  }, [refreshAuth]);

  if (authenticated === null) {
    return <p className="text-muted">Loading…</p>;
  }

  if (!authenticated) {
    return (
      <div className="max-w-md">
        <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-2 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">Watcher</h1>
        <p className="text-muted mb-6">
          Sign in with Trakt to open your watch history, recommendations, and similarity search.
        </p>
        <a
          className="inline-flex items-center justify-center px-5 min-h-11 rounded-lg bg-gradient-to-br from-accent to-accent-dim text-white font-semibold text-sm hover:brightness-110 hover:shadow-[0_0_20px_-4px] hover:shadow-accent/50 hover:no-underline transition-all"
          href={loginHref}
        >
          Log In With Trakt
        </a>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-[1.75rem] font-bold tracking-[-0.04em] mb-1.5 bg-gradient-to-b from-white to-text/70 bg-clip-text text-transparent">Watcher</h1>
      <p className="text-muted mb-8">What do you want to explore?</p>
      <nav className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 max-w-3xl" aria-label="App sections">
        {APP_SECTIONS.map(({ to, label, desc }) => (
          <Link
            key={to}
            to={to}
            className="flex flex-col gap-1.5 p-5 bg-surface border border-border rounded-xl hover:border-accent/30 hover:shadow-[0_8px_30px_-8px] hover:shadow-accent/20 hover:no-underline transition-all duration-200 group"
          >
            <span className="font-semibold text-text group-hover:text-accent transition-colors flex items-center justify-between">
              {label}
              <span className="text-muted group-hover:text-accent translate-x-0 group-hover:translate-x-0.5 transition-all duration-200">→</span>
            </span>
            <span className="text-sm text-muted">{desc}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
}
