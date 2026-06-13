import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Home, Clock, Heart, Star, HelpCircle, Layers, Compass, Users, MessageCircle, Settings,
} from "lucide-react";
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
  const [adminOpen, setAdminOpen] = useState(false);
  const adminRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
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

  const isChatPage = location.pathname === "/chat";

  return (
    <div className="flex flex-col min-h-screen" style={isChatPage ? { height: "100dvh" } : undefined}>
      <header className="sticky top-0 z-10 glass-nav">
        <div className="flex items-center px-4 md:px-6 py-3 max-w-[1320px] mx-auto">
          {/* Logo */}
          <NavLink
            to="/"
            className="flex items-center gap-2 font-bold text-[1.1rem] tracking-tight text-text hover:text-accent transition-colors shrink-0"
          >
            <img
              src="/watcher-logo.png"
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
                <NavLink to="/discover" className={linkClass}>
                  Discover
                </NavLink>
                <NavLink to="/actor" className={linkClass}>
                  Actor Search
                </NavLink>
                <NavLink to="/chat" className={linkClass}>
                  Chat
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
                  <div className="absolute right-0 top-[calc(100%+6px)] glass rounded-2xl shadow-2xl shadow-black/50 overflow-hidden min-w-[160px] z-50">
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

        </div>
      </header>

      {/* Mobile bottom floating nav */}
      {showAppNav && (
        <nav
          className="md:hidden fixed bottom-3 left-3 right-3 z-20 glass-nav rounded-2xl border border-white/10 shadow-2xl shadow-black/40"
          aria-label="Main"
        >
          <div className="flex overflow-x-auto gap-1 px-2 py-2 scrollbar-none">
            {[
              { to: "/", label: "Home", end: true, Icon: Home },
              { to: "/history", label: "History", Icon: Clock },
              { to: "/taste", label: "Taste", Icon: Heart },
              { to: "/recommend", label: "Recs", Icon: Star },
              { to: "/will-like", label: "Will I Like", Icon: HelpCircle },
              { to: "/similar", label: "Similar", Icon: Layers },
              { to: "/discover", label: "Discover", Icon: Compass },
              { to: "/actor", label: "Actors", Icon: Users },
              { to: "/chat", label: "Chat", Icon: MessageCircle },
              { to: "/admin", label: "Admin", Icon: Settings },
            ].map(({ to, label, end, Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  `shrink-0 flex flex-col items-center gap-1 px-3 py-2 rounded-2xl text-[10px] font-medium transition-all whitespace-nowrap ${
                    isActive
                      ? "text-text bg-white/12 border border-white/10"
                      : "text-muted hover:text-text hover:bg-white/6"
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <Icon size={18} strokeWidth={isActive ? 2.5 : 1.75} />
                    {label}
                  </>
                )}
              </NavLink>
            ))}
          </div>
        </nav>
      )}

      <main className={`flex-1 px-4 md:px-6 max-w-[1320px] mx-auto w-full${isChatPage ? " overflow-hidden pb-20 md:pb-0" : " py-7 pb-24 md:pb-7"}`}>
        <Outlet />
      </main>
    </div>
  );
}
