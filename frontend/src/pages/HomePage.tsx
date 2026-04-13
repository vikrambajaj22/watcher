import { useEffect } from "react";
import { Link } from "react-router-dom";
import { getApiBase } from "../api/client";
import { useAuth } from "../contexts/AuthContext";

const APP_SECTIONS = [
  { to: "/history", label: "Watch History" },
  { to: "/visual", label: "Visual Explorer" },
  { to: "/recommend", label: "Recommendations" },
  { to: "/will-like", label: "Will I Like" },
  { to: "/similar", label: "Similar Titles" },
  { to: "/admin", label: "Maintenance" },
] as const;

export function HomePage() {
  const { authenticated, refresh: refreshAuth } = useAuth();
  const loginHref = `${getApiBase()}/auth/trakt/start?from_ui=true`;

  useEffect(() => {
    void refreshAuth();
  }, [refreshAuth]);

  if (authenticated === null) {
    return (
      <div className="page home-simple">
        <p className="muted">Loading…</p>
      </div>
    );
  }

  if (!authenticated) {
    return (
      <div className="page home-simple">
        <h1 className="page-title">Watcher</h1>
        <p className="lede">
          Sign in with Trakt to open your watch history, recommendations, and
          similarity search.
        </p>
        <a className="btn btn-primary" href={loginHref}>
          Log In With Trakt
        </a>
      </div>
    );
  }

  return (
    <div className="page home-simple">
      <h1 className="page-title">Watcher</h1>
      <p className="lede">Choose a section.</p>
      <nav className="home-hub-grid" aria-label="App sections">
        {APP_SECTIONS.map(({ to, label }) => (
          <Link key={to} className="home-quick-link home-hub-link" to={to}>
            {label}
          </Link>
        ))}
      </nav>
    </div>
  );
}
