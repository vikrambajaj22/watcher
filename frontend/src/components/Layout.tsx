import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { apiFetch, getApiBase } from "../api/client";
import { useAuth } from "../contexts/AuthContext";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  isActive
    ? "text-text text-sm font-semibold transition-colors"
    : "text-muted text-sm hover:text-text transition-colors";

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

  const navLinks = (
    <>
      <NavLink to="/" end className={linkClass}>
        Home
      </NavLink>
      {showAppNav && (
        <>
          <NavLink to="/history" className={linkClass}>
            History
          </NavLink>
          <NavLink to="/recommend" className={linkClass}>
            Recommendations
          </NavLink>
          <NavLink to="/will-like" className={linkClass}>
            Will I Like
          </NavLink>
          <NavLink to="/similar" className={linkClass}>
            Similar
          </NavLink>
          <NavLink to="/admin" className={linkClass}>
            Maintenance
          </NavLink>
        </>
      )}
      <a
        href={docsHref}
        target="_blank"
        rel="noreferrer"
        className="text-muted text-sm hover:text-text transition-colors"
      >
        API Docs
      </a>
      {showAppNav && (
        <button
          type="button"
          onClick={() => void logout()}
          className="text-muted text-sm hover:text-text transition-colors cursor-pointer bg-transparent border-0 p-0 font-sans"
        >
          Log Out
        </button>
      )}
    </>
  );

  return (
    <div className="flex flex-col min-h-screen">
      <header className="sticky top-0 z-10 border-b border-border bg-surface/85 backdrop-blur-md">
        <div className="flex items-center gap-4 px-4 md:px-6 py-3 max-w-[1320px] mx-auto">
          <NavLink
            to="/"
            className="flex items-center gap-2 font-bold text-[1.1rem] tracking-tight text-text hover:text-accent transition-colors no-underline! shrink-0"
          >
            <img
              src="/watcher-logo.jpeg"
              alt=""
              height={28}
              className="h-7 w-auto max-w-[7rem] object-contain object-left opacity-90"
              decoding="async"
            />
            Watcher
          </NavLink>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-6 ml-auto" aria-label="Main">
            {navLinks}
          </nav>

          {/* Mobile hamburger */}
          <button
            type="button"
            className="md:hidden ml-auto inline-flex items-center justify-center size-9 rounded-lg border border-border bg-surface text-text cursor-pointer hover:border-muted transition-colors"
            aria-label={navOpen ? "Close menu" : "Open menu"}
            aria-expanded={navOpen}
            onClick={() => setNavOpen((o) => !o)}
          >
            {navOpen ? "✕" : "☰"}
          </button>
        </div>

        {/* Mobile nav */}
        {navOpen && (
          <nav
            className="md:hidden border-t border-border px-4 py-3 flex flex-col gap-1"
            aria-label="Main"
          >
            {[
              { to: "/", label: "Home", end: true },
              ...(showAppNav
                ? [
                    { to: "/history", label: "History" },
                    { to: "/recommend", label: "Recommendations" },
                    { to: "/will-like", label: "Will I Like" },
                    { to: "/similar", label: "Similar" },
                    { to: "/admin", label: "Maintenance" },
                  ]
                : []),
            ].map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `px-2 py-2.5 rounded-lg text-sm transition-colors ${
                    isActive
                      ? "text-text font-semibold bg-accent/8"
                      : "text-muted hover:text-text hover:bg-accent/5"
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
            <a
              href={docsHref}
              target="_blank"
              rel="noreferrer"
              className="px-2 py-2.5 rounded-lg text-sm text-muted hover:text-text hover:bg-accent/5 transition-colors"
            >
              API Docs
            </a>
            {showAppNav && (
              <button
                type="button"
                onClick={() => void logout()}
                className="text-left px-2 py-2.5 rounded-lg text-sm text-muted hover:text-text hover:bg-accent/5 transition-colors cursor-pointer bg-transparent border-0 font-sans w-full"
              >
                Log Out
              </button>
            )}
          </nav>
        )}
      </header>

      <main className="flex-1 px-4 md:px-6 py-7 max-w-[1320px] mx-auto w-full">
        <Outlet />
      </main>
    </div>
  );
}
