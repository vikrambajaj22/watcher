# Watcher UI (React)

Single-user SPA — the supported Watcher UI. Dev server proxies API calls to FastAPI on port **8080**.

Routes: `/` (home), `/history`, `/visual`, `/recommend`, `/will-like`, `/similar`, `/admin` (shown as **Maintenance** in the nav).

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:8501** (default; match `UI_BASE_URL` and your Trakt app).

## Environment

| Variable | When |
|----------|------|
| (none) | **Dev** — uses Vite proxy `/api-proxy` → `http://127.0.0.1:8080` |
| `VITE_API_BASE_URL` | **Production build** — full API origin, e.g. `https://your-backend.run.app` |
| `VITE_ADMIN_API_KEY` | Optional dev/build setting only: same secret as the API’s admin password if you use one. Not part of the in-app UI. |

Example production build:

```bash
VITE_API_BASE_URL=https://watcher-backend-xxxxx.run.app \
VITE_ADMIN_API_KEY=your-secret \
npm run build
```

Serve the `dist/` folder from any static host, or build the included **`Dockerfile`** / **`cloudbuild.yaml`** for nginx on Cloud Run (see [DEPLOYMENT.md](../DEPLOYMENT.md)).

## Backend CORS

The API allows browser origins from `WATCHER_CORS_ORIGINS` (comma-separated). Defaults include `http://localhost:8501`. For production, set e.g.:

```bash
WATCHER_CORS_ORIGINS=https://your-ui-host.run.app,https://your-ui.pages.dev
```

## Trakt OAuth

Set backend **`UI_BASE_URL`** to this UI’s public URL (e.g. `http://localhost:8501` in dev) so the Trakt callback redirects back correctly after login.

## Notes

- When the API uses an optional access key (`ADMIN_API_KEY`), the frontend needs the matching `VITE_ADMIN_API_KEY` in dev/build if the browser should call those routes. See [DEVELOPMENT.md](../DEVELOPMENT.md) for the full endpoint list and troubleshooting.
- Production bundle includes Recharts (~500kB JS); use code-splitting later if you need a smaller first load.
