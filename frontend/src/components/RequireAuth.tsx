import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export function RequireAuth() {
  const { authenticated } = useAuth();
  const location = useLocation();

  if (authenticated === null) {
    return (
      <div className="page">
        <p className="muted">Checking Trakt Sign-In…</p>
      </div>
    );
  }

  if (!authenticated) {
    return <Navigate to="/" replace state={{ from: location.pathname }} />;
  }

  return <Outlet />;
}
