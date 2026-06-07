import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { apiFetch, getApiBase } from "../api/client";
import { useAuth } from "../contexts/AuthContext";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  isActive
    ? "text-text text-sm font-medium px-3 py-1.5 rounded-full bg-white/8 border border-white/6 transition-all"
    : "text-muted text-sm px-3 py-1.5 rounded-full hover:text-text hover:bg-white/5 transition-all";

export function Layout() {
  const docsHref = `${getApiBase()}/docs`;
  const { authenticated, refresh } = useAuth();
  const showAppNav = authenticated === true;
  const location = useLocation();
  const [navOpen, setNavOpen] = useState(false);
  const [adminOpen, setAdminOpen] = useState(false);
  const adminRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setNavOpen(false);
    setAdminOpen(false);
  }, [location.pathname, location.search]);

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (adminRef.current && !adminRef.current.contains(e.target as Node)) {
        setAdminOpen(false);
      }
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

  async function logout() {
    try {
      await apiFetch("/auth/logout");
    } catch {
      /* still refresh session */
    }
    await refresh();
  }

  const isAdminActive =
    location.pathname === "/admin" || location.pathname.startsWith("/admin");

  return (
    <div className="flex flex-col min-h-screen">
      <header className="sticky top-0 z-10 border-b border-border/60 bg-bg/80 backdrop-blur-xl">
        <div className="flex items-center px-4 md:px-6 py-3 max-w-[1320px] mx-auto">
          {/* Logo */}
          <NavLink
            to="/"
            className="flex items-center gap-2 font-bold text-[1.1rem] tracking-tight text-text hover:text-accent transition-colors shrink-0"
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

          {/* Desktop nav — centered */}
          <nav className="hidden md:flex items-center gap-1 flex-1 justify-center" aria-label="Main">
            <NavLink to="/" end className={linkClass}>
              Home
            </NavLink>
            {showAppNav && (
              <>
                <NavLink to="/history" className={linkClass}>
                  History
                </NavLink>
                <NavLink to="/taste" className={linkClass}>
                  Taste Profile
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
              </>
            )}
          </nav>

          {/* Desktop right — Admin dropdown + Logout */}
          <div className="hidden md:flex items-center gap-1 shrink-0">
            {showAppNav && (
              <div className="relative" ref={adminRef}>
                <button
                  type="button"
                  onClick={() => setAdminOpen((o) => !o)}
                  className={`text-sm px-3 py-1.5 rounded-full transition-all cursor-pointer bg-transparent border font-sans ${
                    isAdminActive || adminOpen
                      ? "text-text font-medium bg-white/8 border-white/6"
                      : "text-muted border-transparent hover:text-text hover:bg-white/5"
                  }`}
                >
                  Admin
                </button>
                {adminOpen && (
                  <div className="absolute right-0 top-[calc(100%+6px)] bg-surface border border-border rounded-xl shadow-2xl shadow-black/50 overflow-hidden min-w-[160px] z-50">
                    <NavLink
                      to="/admin"
                      className={({ isActive }) =>
                        `block px-4 py-2.5 text-sm transition-colors ${
                          isActive
                            ? "text-text font-medium bg-accent/8"
                            : "text-muted hover:text-text hover:bg-white/5"
                        }`
                      }
                      onClick={() => setAdminOpen(false)}
                    >
                      Maintenance
                    </NavLink>
                    <a
                      href={docsHref}
                      target="_blank"
                      rel="noreferrer"
                      className="block px-4 py-2.5 text-sm text-muted hover:text-text hover:bg-white/5 transition-colors border-t border-border"
                      onClick={() => setAdminOpen(false)}
                    >
                      API Docs
                    </a>
                  </div>
                )}
              </div>
            )}
            {showAppNav && (
              <button
                type="button"
                onClick={() => void logout()}
                className="text-muted text-sm px-3 py-1.5 rounded-full hover:text-text hover:bg-white/5 transition-all cursor-pointer bg-transparent border-0 font-sans"
              >
                Log Out
              </button>
            )}
          </div>

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
                    { to: "/taste", label: "Taste Profile" },
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
