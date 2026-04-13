# Development & operator reference

Notes for people running or hacking on Watcher: how the React UI talks to the API, optional locking of routes, and where to look when something fails.

## Optional API lock (`ADMIN_API_KEY`)

If the API process has `ADMIN_API_KEY` set in its environment, routes that change data, run sync jobs, touch the vector index, or expose MCP-style helpers expect an **`X-API-Key`** HTTP header with the **same** secret.

- **`./start.sh`** loads the repo-root `.env` for the API process.
- **React (Vite)** does not read the API’s `.env`. For local dev, create **`frontend/.env`** or **`frontend/.env.local`** with:
  ```env
  VITE_ADMIN_API_KEY=same-secret-as-ADMIN_API_KEY
  ```
  The dev client injects that into `X-API-Key` on outgoing requests (see `frontend/src/api/client.ts`).
- **Production** React builds: set `VITE_ADMIN_API_KEY` at **build** time if the deployed API uses the lock and the browser must call those routes. Otherwise omit it and keep those routes server-side only.

Also set **`WATCHER_CORS_ORIGINS`** on the API to include your UI origin when the UI is not served from the same host as the API.

Further env detail: [README.md](README.md) (Environment, Quick Start) and [frontend/README.md](frontend/README.md).

## React SPA routes → API (summary)

| UI area | Typical methods | Paths |
|--------|-----------------|--------|
| Home | GET | `/health`, `/auth/status`, `/admin/sync/status` |
| Watch history | GET | `/history` (query params as used in app) |
| Watch history | POST | `/admin/sync/trakt`, `/admin/clear-history-cache` |
| Recommendations | POST | `/recommend/{all\|movie\|tv}` |
| Will I like | POST | `/mcp/will-like` |
| Similar titles | POST | `/mcp/knn` |
| Similar titles (metadata label) | GET | `/admin/tmdb/{id}` |
| Visual explorer | GET | `/visualize/clusters` |
| Maintenance (admin UI) | GET/POST | `/admin/*` (sync jobs, FAISS, embed, cache clear) |

OpenAPI / interactive docs: `http://localhost:8080/docs` (or `/redoc`) when the API is running.

**From the React dev server (`http://localhost:8501`):** the **API Docs** nav link opens `/api-proxy/docs`. Vite proxies that to the API, but Swagger loads the spec from **`/openapi.json`** on the **same origin** (8501). `vite.config.ts` therefore proxies `/openapi.json` to the backend as well; without it, Swagger shows an invalid-schema error because it receives the SPA’s `index.html` instead of JSON.

## Maintenance UI

The React **Maintenance** screen exposes the same sync / index / cache actions as the FastAPI routes below.

## Related docs

- [README.md](README.md) — features, env vars, embedding/FAISS overview, curl examples  
- [DEPLOYMENT.md](DEPLOYMENT.md) — production layout, secrets, GCP  
- [PLAN.md](PLAN.md) — roadmap and technical debt