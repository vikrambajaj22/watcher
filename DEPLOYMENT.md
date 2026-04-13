# Watcher Full Deployment on GCP

This guide covers **full deployment of the Watcher app** on Google Cloud Platform: MongoDB on an e2-micro VM, data
transfer, **FastAPI on Cloud Run**, and the **React SPA** (static build — either your own static host or the optional
nginx image on Cloud Run described below).

---

## Set Up GCP Project

You can either export environment variables manually as shown below, or (recommended) keep them in a `**.env.gcp`** file at the repo root (gitignored), then run `**source .env.gcp**` in every new shell before `gcloud` / build commands so `PROJECT_ID`, API keys, FAISS settings, and `**ADMIN_API_KEY**` are defined.

`**ADMIN_API_KEY`:** Generate it **once** (e.g. `openssl rand -hex 32`), put the value in `.env.gcp` as `ADMIN_API_KEY=...`, and **reuse it** on the backend and in **every new React production build** as `VITE_ADMIN_API_KEY` (same string). If you rotate the key, redeploy the backend **and** rebuild/redeploy the UI so the baked-in `VITE_ADMIN_API_KEY` matches; otherwise the browser will get `403` on protected routes.

### Create a new GCP project:

```bash
gcloud projects create watcher-project --name="Watcher"
gcloud config set project <PROJECT_ID>  # get <PROJECT_ID> from the created project on GCP Console
```

### Enable required services:

```bash
gcloud services enable compute.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Create VM for MongoDB

### Create an e2-micro VM:

```bash
gcloud compute instances create watcher-mongo \
 --zone=us-central1-a \
 --machine-type=e2-micro \
 --tags=mongo
```

### Open firewall for MongoDB (port 27017):

```bash
gcloud compute firewall-rules create allow-mongo \
--allow=tcp:27017 \
--target-tags=mongo \
--description="Allow MongoDB access"
```

### SSH into the VM:

```bash
gcloud compute ssh watcher-mongo --zone=us-central1-a
```

### Install MongoDB on the VM:

```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
```

#### Add swap to reduce OOM kills on e2-micro

On many small GCE VMs (including e2-micro), there is no swap configured by default. When MongoDB plus the OS exceed physical RAM, the kernel may kill `mongod`. Adding a small swapfile uses disk as a safety valve and greatly reduces OOM crashes (at the cost of slower performance under heavy memory pressure).

Run these commands **on the VM shell (not inside `mongosh`)**:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

If `free -h` shows a non-zero `Swap` line, swap is enabled and will persist across reboots.

#### Configure MongoDB to use less memory (optional for e2-micro) for cache size:

In the file `/etc/mongod.conf`, add the following under `storage` (use `sudo` if needed):

```yaml
wiredTiger:
  engineConfig:
    cacheSizeGB: 0.3
```

#### Enable journaling (recommended):

In the same file `/etc/mongod.conf`, ensure the following under `storage`:

```yaml
journal:
  enabled: true
```

Journaling helps maintain data integrity in case of crashes, so crashes don’t wipe out recent writes.

#### Persist MongoDB data on disk (override auto-deletion on VM stop):

```bash
gcloud compute instances set-disk-auto-delete watcher-mongo \
--zone=us-central1-a \
--disk=watcher-mongo \
--no-auto-delete
```

#### Configure MongoDB to bind to all IP addresses:

In the same file `/etc/mongod.conf`, modify the `bindIp` under `net` to:

```yaml
net:
  port: 27017
  bindIp: 0.0.0.0
```

### Start MongoDB:

```bash
sudo systemctl start mongod
sudo systemctl enable mongod
```

### Confirm MongoDB is running:

```bash
mongo --eval 'db.runCommand({ connectionStatus: 1 })'
```

### Create MongoDB User

- Access MongoDB shell:

```bash
mongosh
```

- Switch to `admin` database and create user:

```bash
use admin
db.createUser({
  user: "<GCP_MONGO_DB_USER>",
  pwd: "<GCP_MONGO_DB_PASSWORD>",
  roles: [
    { role: "readWrite", db: "watcher" },
    { role: "userAdminAnyDatabase", db: "admin" }
  ]
})
```

- Exit shell:

```bash
exit
```

### Restart MongoDB to apply changes:

```bash
sudo systemctl restart mongod
```

### Transfer Local MongoDB Dumps to GCP VM

Pick an approach based on **disk and RAM** on the e2-micro host (small disk benefits from smaller dumps).

#### Option A — `mongodump` / `mongorestore` (full documents + indexes)

Use when you want Mongo’s native dump layout and **no field pruning** (larger files and transfer).

```bash
# On your local machine (replace URI with your local connection string)
mongodump --uri="mongodb://localhost:27017/watcher" --out=mongo_dump_artifact
gcloud compute scp --recurse mongo_dump_artifact watcher-mongo:~/ --zone=us-central1-a
```

On the VM, restore with the **same user you created earlier** (authSource is `admin`):

```bash
mongorestore --uri="mongodb://<GCP_MONGO_DB_USER>:<GCP_MONGO_DB_PASSWORD>@127.0.0.1:27017/watcher?authSource=admin" \
  --drop \
  mongo_dump_artifact/watcher
```

Skip `--drop` if you intend to merge instead of replace (advanced).

#### Option B — `tools/mongo_local_dump_export.py` (pruned `tmdb_metadata`, smaller BSON)

This is a **deliberate tradeoff** for tiny VMs: only the fields listed in `TMDB_FIELDS` in that script are kept for `tmdb_metadata` (plus pruned `credits`), which cuts Mongo size and transfer time. Other collections are exported whole. That set is meant to match what the app uses for recommendations, embeddings text, and UI; if you add features that read new TMDB fields, **extend `TMDB_FIELDS`** (or run once with `MONGO_DUMP_PRUNE_TMDB=0` for a full metadata export).

The script writes one `.bson` file per collection (not a `mongodump` directory tree). **ObjectIds are kept by default**; set `MONGO_DUMP_STRIP_OBJECT_ID=1` only if you intentionally want smaller files and new ids on restore.

```bash
# From repo root (default: pruned tmdb_metadata — good for e2-micro)
python tools/mongo_local_dump_export.py
gcloud compute scp --recurse mongo_dumps watcher-mongo:~/ --zone=us-central1-a
```

On the VM (after SSH), restore **each** collection file with the authenticated URI:

```bash
# No need for export — the shell substitutes $MONGO_URI when each command runs.
MONGO_URI='mongodb://<GCP_MONGO_DB_USER>:<GCP_MONGO_DB_PASSWORD>@127.0.0.1:27017/watcher?authSource=admin'

mongorestore --uri="$MONGO_URI" --drop -d watcher -c tmdb_metadata ~/mongo_dumps/tmdb_metadata.bson
mongorestore --uri="$MONGO_URI" --drop -d watcher -c watch_history ~/mongo_dumps/watch_history.bson
# Repeat for every *.bson in ~/mongo_dumps (-c is the collection name = file stem).
```

**Common pitfalls**

- **No auth on `mongorestore`:** If you enabled a user in `admin`, localhost connections still need `--uri=...authSource=admin` with user/password.
- **Wrong working directory:** After `scp`, files live at `~/mongo_dumps/...`. Either `cd ~` or pass absolute paths.
- **Flat `.bson` files:** The Python exporter produces one file per collection; restore each with `-d watcher -c <collection>` as above. A `mongodump` directory is restored differently (Option A).

Verify:

```bash
mongosh "$MONGO_URI"
> db.getSiblingDB("watcher").tmdb_metadata.countDocuments()
```

When finished verifying, **exit `mongosh`** (e.g. type `exit` or press Ctrl+D) so you are back at the normal shell. Then remove the dump from the VM and restart MongoDB:

```bash
rm -rf ~/mongo_dumps ~/mongo_dump_artifact
sudo systemctl restart mongod
```

(Delete only the path you actually used — `mongo_dumps` from the Python exporter, or `mongo_dump_artifact` from `mongodump`.)

## Build and Push Backend Docker Image

From the **repository root**, load deploy variables (including a **stable** `ADMIN_API_KEY` from `.env.gcp`):

```bash
cd /path/to/watcher
source .env.gcp
```

### Build image on GCP Cloud Build and push to Container Registry

Still at the **repository root** (where the backend `Dockerfile` lives):

```bash
export BACKEND_IMAGE="us-central1-docker.pkg.dev/$PROJECT_ID/watcher/api"
export REGION=us-central1
export MONGO_VM_IP="<VM_EXTERNAL_IP>"  # replace with your mongo VMs external IP
gcloud builds submit . --tag $BACKEND_IMAGE
```

### Deploy Backend to Cloud Run

Deploy with all required environment variables (after `source .env.gcp`):

**Note:** The backend requires 8Gi memory to accommodate:

- FAISS index loaded into memory
- Embedding model (sentence-transformers)
- Application overhead

```bash
source .env.gcp
gcloud run deploy watcher-backend \
--image $BACKEND_IMAGE \
--region $REGION \
--platform managed \
--execution-environment gen2 \
--allow-unauthenticated \
--memory 8Gi \
--cpu 2 \
--concurrency 1 \
--timeout 900 \
--add-volume=name=faiss-data,type=cloud-storage,bucket=$FAISS_BUCKET \
--add-volume-mount=volume=faiss-data,mount-path=/mnt/faiss \
--set-env-vars \
MONGODB_URI="mongodb://$GCP_MONGO_DB_USER:$GCP_MONGO_DB_PASSWORD@$MONGO_VM_IP:27017/watcher?authSource=admin",\
EMBED_DEVICE=cpu,\
FAISS_USE_GPU=false,\
FAISS_SOURCE=$FAISS_SOURCE,\
FAISS_BUCKET=$FAISS_BUCKET,\
FAISS_PREFIX=$FAISS_PREFIX,\
FAISS_MOUNT_PATH=$FAISS_MOUNT_PATH,\
FAISS_PERSIST_DIR=$FAISS_PERSIST_DIR,\
OPENAI_API_KEY=$OPENAI_API_KEY,\
TMDB_API_KEY=$TMDB_API_KEY,\
TRAKT_CLIENT_ID=$TRAKT_CLIENT_ID,\
TRAKT_CLIENT_SECRET=$TRAKT_CLIENT_SECRET,\
TRAKT_REDIRECT_URI=$TRAKT_REDIRECT_URI,\
UI_BASE_URL=$UI_BASE_URL,\
ADMIN_API_KEY=$ADMIN_API_KEY
```

`**ADMIN_API_KEY` (recommended for production):** If set to a non-empty secret, all admin, MCP, sync job, `/auth/logout`, and `/visualize/*` routes require the HTTP header `X-API-Key: <same value>`. If unset or empty, those routes accept requests without a key (convenient for local dev only). The **React** UI needs the same secret at build time as `VITE_ADMIN_API_KEY` when the browser must call those routes—see [DEVELOPMENT.md](DEVELOPMENT.md) and the UI deploy step below. Store the value in `**.env.gcp`** and keep it unchanged across deploys unless you intentionally rotate it (then update **both** backend and UI artifacts to the new value).

The `?authSource=admin` parameter in the `MONGODB_URI` is necessary since we created the user in the `admin` database.

Once `**watcher-backend`** is deployed, point Trakt at the **API** callback (same host as `BACKEND_URL`):

```bash
source .env.gcp
export REGION="${REGION:-us-central1}"
export BACKEND_URL=$(gcloud run services describe watcher-backend --region="$REGION" --format='value(status.url)')

gcloud run services update watcher-backend \
  --region "$REGION" \
  --update-env-vars TRAKT_REDIRECT_URI="${BACKEND_URL}/auth/trakt/callback"
```

Register that exact URL in your **Trakt application** settings. After the React app is deployed (next section), set `**UI_BASE_URL`** and `**WATCHER_CORS_ORIGINS**` in **Wire backend ↔ SPA**.

Use `**--update-env-vars`** for partial changes. `**--set-env-vars` replaces the entire environment** on the service and can drop `MONGODB_URI`, `ADMIN_API_KEY`, FAISS settings, etc., if you omit them.

For the FAISS setup, please refer to the FAISS section below. Use the required environment variables when deploying the backend. To avoid cold-start download latency, use **Option A (Mount GCS bucket)** in the FAISS section and add `--execution-environment gen2`, `--add-volume`, `--add-volume-mount`, and `FAISS_MOUNT_PATH=/mnt/faiss/v1` to the deploy command.

## Build and deploy the React UI (static SPA)

The UI lives in `**frontend/`**. For production, `**VITE_API_BASE_URL**` must be the **public HTTPS origin of the API** (your Cloud Run backend URL, **no trailing slash**). `**VITE_ADMIN_API_KEY`** must equal `**ADMIN_API_KEY**` when the API enforces the admin key; both are **baked in at build time** (there is no runtime `API_BASE_URL` env on the client).

From the repo root (after the backend is deployed):

```bash
source .env.gcp
export REGION="${REGION:-us-central1}"
export BACKEND_URL=$(gcloud run services describe watcher-backend --region="$REGION" --format='value(status.url)')
```

### Option A — Build locally, host `dist` anywhere

Use this when you prefer Firebase / Cloudflare Pages / Netlify / S3+CDN, or any static file host.

```bash
cd frontend
npm ci
VITE_API_BASE_URL="$BACKEND_URL" VITE_ADMIN_API_KEY="$ADMIN_API_KEY" npm run build
cd ..
```

Upload the contents of `**frontend/dist/**` to your host. Your site’s public URL is `**UI_PUBLIC_URL**` — use it in **Wire backend ↔ SPA** for `**UI_BASE_URL`** and `**WATCHER_CORS_ORIGINS**`.

Example (from **repo root** after `npm run build`; public bucket website — adjust bucket name, IAM, and load balancer/DNS to your standards):

```bash
export UI_BUCKET=your-watcher-ui-bucket
gsutil -m rsync -r -d frontend/dist gs://$UI_BUCKET/
# Point DNS or a load balancer at the bucket’s website endpoint; that HTTPS URL is UI_PUBLIC_URL.
```

### Option B — Cloud Build + Cloud Run (nginx serving the SPA)

The repo includes `**frontend/Dockerfile**` (multi-stage: `npm run build` + nginx on port **8080**) and `**frontend/cloudbuild.yaml`**. Build args pass `VITE_*` into the image on GCP (avoid relying on `dist/` in git).

```bash
source .env.gcp
export REGION="${REGION:-us-central1}"
export BACKEND_URL=$(gcloud run services describe watcher-backend --region="$REGION" --format='value(status.url)')
export UI_IMAGE="us-central1-docker.pkg.dev/$PROJECT_ID/watcher/ui"

gcloud builds submit ./frontend \
  --config=./frontend/cloudbuild.yaml \
  --substitutions=_UI_IMAGE="$UI_IMAGE",_VITE_API_BASE_URL="$BACKEND_URL",_VITE_ADMIN_API_KEY="$ADMIN_API_KEY"
```

Deploy the UI service (separate from the API; nginx listens on **8080**):

```bash
gcloud run deploy watcher-ui \
  --image "$UI_IMAGE" \
  --region "$REGION" \
  --platform managed \
  --execution-environment gen2 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 5 \
  --port 8080
```

Set `**UI_PUBLIC_URL**` to the HTTPS URL where the SPA is served (Cloud Run URL for `**watcher-ui**`, or your static host).

### Wire backend ↔ SPA

The browser loads the SPA from `**UI_PUBLIC_URL**` and calls the API at `**BACKEND_URL**`. Set the backend env so Trakt redirects users back to the SPA and CORS allows the browser origin:

```bash
source .env.gcp
export REGION="${REGION:-us-central1}"
export BACKEND_URL=$(gcloud run services describe watcher-backend --region="$REGION" --format='value(status.url)')
export UI_PUBLIC_URL="https://YOUR-SPA-HOST"   # no trailing slash

gcloud run services update watcher-backend \
  --region "$REGION" \
  --update-env-vars \
UI_BASE_URL="$UI_PUBLIC_URL",\
WATCHER_CORS_ORIGINS="$UI_PUBLIC_URL"
```

Use `**WATCHER_CORS_ORIGINS**` exactly as the browser sees the UI origin (scheme + host, no path). For several origins, use a comma-separated list (no spaces, or quote the value). If you do not use `**ADMIN_API_KEY**` on the API, omit `**VITE_ADMIN_API_KEY**` when building the frontend.

### After the UI is live

1. **Trakt developer app:** Allowed origins / redirect settings must include `**UI_PUBLIC_URL`** where required by Trakt.
2. **Smoke test:** Open `**UI_PUBLIC_URL`**, sign in with Trakt, and confirm API calls succeed (browser devtools → Network).

See also [frontend/README.md](frontend/README.md) and [DEVELOPMENT.md](DEVELOPMENT.md) for env details.

## FAISS for TMDB

FAISS is used for similarity search, candidate generation for recommendations, etc.
To enable FAISS functionality, you need to ensure that the FAISS index and precomputed embeddings are accessible to the backend service.
We do this using local embedding computation and upload to a GCS bucket `watcher-faiss`.

This approach avoids:

- embedding data in Docker images
- MongoDB size limits by keeping embeddings out of the database
- rebuilding images when the index changes

**FAISS artifacts**

- `tmdb.index` – FAISS index
- `labels.npy` – label → TMDB id mapping
- `vecs.npy` – optional raw vectors

**Runtime behavior**

- Local development → load FAISS from local filesystem
- Cloud Run (download) → download FAISS from GCS into `FAISS_PERSIST_DIR` if missing, then load into memory (adds cold-start latency)
- Cloud Run (mount) → mount GCS bucket as a volume; backend reads index directly (no download, faster cold starts)
- **Memory requirement:** The backend needs at least 8Gi memory to load the FAISS index and embedding model into memory

The backend uses these environment variables:

```bash
FAISS_SOURCE=gcs  # or unset for local (defaults to ./faiss_index/ if unset)
FAISS_BUCKET=watcher-faiss
FAISS_PREFIX=v1  # versioned folder in bucket
FAISS_MOUNT_PATH=  # optional: when set (e.g. /mnt/faiss/v1), read from this path instead of downloading (use with Cloud Run volume mount)
FAISS_PERSIST_DIR=/var/lib/faiss  # optional: where downloads are cached on-disk when FAISS_SOURCE=gcs and FAISS_MOUNT_PATH is unset
```

- **Recommendation & precedence (explicit):**
  - `FAISS_MOUNT_PATH` takes precedence over download mode: if `FAISS_MOUNT_PATH` is set (non-empty), the app will read `tmdb.index`, `labels.npy`, `vecs.npy`, and `sidecar_meta.json` directly from that filesystem path and will not perform a GCS download. Use this when you have mounted the bucket (see Option A).
  - Mounting the bucket (Option A) is the recommended production setup: it avoids per-instance downloads, greatly reduces cold-start latency, and presents a single canonical copy of the FAISS artifacts. Mounting requires Cloud Run **Gen2** and the `--add-volume` / `--add-volume-mount` flags.
  - If you cannot mount (Option B — download), explicitly set `FAISS_PERSIST_DIR` to a known writable directory (for example `/tmp/faiss`) to avoid unexpected fallbacks from the default `/var/lib/faiss`. Example deploy using download mode:
  - If you use download mode and care about user-facing cold starts, consider keeping at least one pre-warmed instance (`min-instances`) or using a mount.
  - Ensure the Cloud Run service account has `storage.objectViewer` on the FAISS bucket in either mode (download or mount).

### Local Embedding Computation

**Optional**: Run a TMDB sync to get any new items into the local MongoDB - FAISS embeddings are computed during the sync.

To recompute embeddings locally, run `./start.sh`, open the React app **Maintenance** page, and run a full embedding / index rebuild (Embeddings & index tab). In production you can use the same controls in the deployed SPA (with `ADMIN_API_KEY` / `VITE_ADMIN_API_KEY` if enabled) or call the HTTP endpoints directly.

### Upload FAISS Index to GCS Bucket

#### Create a GCS bucket:

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1
export FAISS_BUCKET=watcher-faiss
```

```bash
gsutil mb \
-p $PROJECT_ID \
-l $REGION \
-c STANDARD \
gs://$FAISS_BUCKET
```

#### Upload FAISS artifacts:

```bash
gsutil -m cp \
tmdb.index \
labels.npy \
vecs.npy \
sidecar_meta.json \
gs://$FAISS_BUCKET/v1/
```

#### Get IAM permissions for Cloud Run service account:

```bash
gcloud projects describe $PROJECT_ID --format="value(projectNumber)"
export PROJECT_NUMBER=<PREV_COMMAND_OUTPUT>
export RUN_SA="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"
gsutil iam ch \
serviceAccount:$RUN_SA:objectViewer \
gs://$FAISS_BUCKET
```

#### Option A: Mount GCS bucket (recommended — no download, faster cold starts)

Mount the `watcher-faiss` bucket so the backend reads FAISS files directly from the volume instead of downloading at startup. Requires Cloud Run **Gen 2** (`--execution-environment gen2`).

**Deploy (or update) the backend with a volume and env:**

Deploy the backend using the gen2 execution environment, making sure to include all mount-related flags and environment variables. If you are deploying for the first time, include the `--add-volume` and `--add-volume-mount` flags in the deploy command as shown in the backend deployment command previously.

If the service already exists, and you only need to add the volume and mount:

The bucket root is mounted at `/mnt/faiss`, so with artifacts under `v1/` in the bucket, the app reads from `/mnt/faiss/v1` (set `FAISS_MOUNT_PATH=/mnt/faiss/v1`). On cold start the backend loads the index from the mount immediately — no GCS download step.

#### Option B: Download from GCS at startup (no volume mount)

```bash
gcloud run services update watcher-backend \
--region $REGION \
--set-env-vars FAISS_SOURCE=gcs,FAISS_BUCKET=$FAISS_BUCKET,FAISS_PREFIX=v1,FAISS_PERSIST_DIR=/var/lib/faiss
```

On the next cold start (or first start on a new instance), the backend will:

- Download FAISS files from GCS
- Store them in `FAISS_PERSIST_DIR` (defaults to `/var/lib/faiss`; if not writable it falls back to `/tmp/faiss`)
- Load the index into memory

Notes:

- If `FAISS_PERSIST_DIR` is writable, subsequent reloads within the same Cloud Run instance won’t re-download the 2GB index (files are cached on disk and the process also keeps an in-memory `_cached_index`).
- To fully avoid cold-start download latency across new instances, use **Option A (volume mount)**.

#### Keeping FAISS Updated

Whenever you need to update the FAISS index or embeddings (e.g., after adding new items), repeat the upload step to the GCS bucket (e.g. under a new version folder):

```bash
gsutil -m cp \
tmdb.index \
labels.npy \
vecs.npy \
sidecar_meta.json \
gs://$FAISS_BUCKET/v2/
```

Then update the backend to use the new version:

- **Option A (volume mount):** set `FAISS_MOUNT_PATH=/mnt/faiss/v2` and redeploy/update the service.
- **Option B (download):** set `FAISS_PREFIX=v2` and update the service.

```bash
gcloud run services update watcher-backend \
--region $REGION \
--set-env-vars FAISS_MOUNT_PATH=/mnt/faiss/v2
# or for Option B: --set-env-vars FAISS_PREFIX=v2
```

This will trigger a restart of the backend service, which will load the updated FAISS index on app start before serving requests, without requiring a new backend image build.

## Verify deployment

- **Backend:** `curl -sS "$BACKEND_URL/health"` should return JSON with a healthy status.
- **UI:** Open `**UI_PUBLIC_URL`** (your static site or `watcher-ui` Cloud Run URL). Sign in with Trakt and load Watch history or Recommendations.
- If the SPA loads but API calls fail with CORS errors, fix `**WATCHER_CORS_ORIGINS**` on the backend (exact browser origin, no path).

## Future Improvements

- Stateless UI & horizontal scaling: The primary UI is a static/React SPA backed by APIs; ensure session-sensitive flows remain server-side so multiple UI or API instances stay safe to scale.
- FAISS Index Lifecycle Management: Automate FAISS index rebuilds and versioning when new TMDB items arrive, with background jobs and atomic index swaps to avoid downtime.
- GPU-Backed Recommendations: Evaluate deploying the backend on GPU-enabled infrastructure for faster embedding search once FAISS usage and query volume increase.
- Improved Auth & Secrets Management: Migrate all secrets (API keys, OAuth credentials) to Secret Manager and tighten IAM roles for least-privilege access.
- Observability & Cost Controls: Add structured metrics (latency, cache hit rate, index load time) and alerts to better understand performance and manage GCP costs as usage grows.

