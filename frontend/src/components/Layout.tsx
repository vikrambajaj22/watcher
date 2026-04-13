import { useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { apiFetch, getApiBase } from "../api/client";
import { useAuth } from "../contexts/AuthContext";

export function Layout() {
  const docsHref = `${getApiBase()}/docs`;
  const { authenticated, refresh } = useAuth();
  const showAppNav = authenticated === true;
  const location = useLocation();
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname, location.search]);

  async function logout() {
    try {
      await apiFetch("/auth/logout");
    } catch {
      /* still refresh session */
    }
    await refresh();
  }

  return (
    <div className="app-shell">
      <header className="top-nav">
        <Link to="/" className="brand">
          <img
            src="/watcher-logo.jpeg"
            alt=""
            className="brand-icon"
            height={32}
            decoding="async"
          />
          Watcher
        </Link>
        <button
          type="button"
          className={`nav-menu-btn${navOpen ? " is-open" : ""}`}
          aria-label={navOpen ? "Close menu" : "Open menu"}
          aria-expanded={navOpen}
          aria-controls="site-nav"
          onClick={() => setNavOpen((o) => !o)}
        />
        <nav
          className={`nav-links${navOpen ? " is-open" : ""}`}
          id="site-nav"
          aria-label="Main"
        >
          <Link to="/">Home</Link>
          {showAppNav && (
            <>
              <Link to="/history">Watch History</Link>
              <Link to="/visual">Visual Explorer</Link>
              <Link to="/recommend">Recommendations</Link>
              <Link to="/will-like">Will I Like</Link>
              <Link to="/similar">Similar Titles</Link>
              <Link to="/admin">Maintenance</Link>
            </>
          )}
          <a href={docsHref} target="_blank" rel="noreferrer">
            API Docs
          </a>
          {showAppNav && (
            <button
              type="button"
              className="nav-action"
              onClick={() => void logout()}
            >
              Log Out
            </button>
          )}
        </nav>
      </header>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
