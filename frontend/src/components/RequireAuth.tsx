import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export function RequireAuth() {
  const { authenticated } = useAuth();
  const location = useLocation();

  if (authenticated === null) {
    return <p className="text-muted">Checking sign-in…</p>;
  }

  if (!authenticated) {
    return <Navigate to="/" replace state={{ from: location.pathname }} />;
  }

  return <Outlet />;
}
