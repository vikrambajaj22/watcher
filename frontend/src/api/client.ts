/**
 * API base: dev uses Vite proxy; production uses VITE_API_BASE_URL from build env.
 */
export function getApiBase(): string {
  if (import.meta.env.DEV) {
    return "/api-proxy";
  }
  const u = String(import.meta.env.VITE_API_BASE_URL ?? "")
    .replace(/^\uFEFF/, "")
    .trim()
    .replace(/\/$/, "");
  return u;
}

export function apiUrl(path: string): string {
  const base = getApiBase();
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${base}${p}`;
}

function normalizeAdminKey(): string {
  const raw = import.meta.env.VITE_ADMIN_API_KEY;
  if (raw == null || typeof raw !== "string") return "";
  // CRLF / BOM break Header validation in WebKit ("The string did not match the expected pattern")
  return raw.replace(/^\uFEFF/, "").trim();
}

function adminHeaders(): HeadersInit {
  const key = normalizeAdminKey();
  if (key) {
    return { "X-API-Key": key };
  }
  return {};
}

export async function apiFetch(
  path: string,
  init: RequestInit = {}
): Promise<Response> {
  const headers = new Headers(init.headers);
  const admin = adminHeaders();
  for (const [k, v] of Object.entries(admin)) {
    if (!headers.has(k)) {
      headers.set(k, v);
    }
  }
  return fetch(apiUrl(path), { ...init, headers });
}
