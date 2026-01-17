# Watcher Full Deployment on GCP

This guide covers **full deployment of the Watcher app** on Google Cloud Platform, including MongoDB setup on an
e2-micro VM, pushing data from local to GCP, and deploying backend and UI containers to Cloud Run.

---

## Set Up GCP Project

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

#### On local machine, locate your dump folder (created by running `python tools/mongo_local_dump_export.py`):

```
mongo_dumps/
  └─ tmdb_metadata.bson ...
```

#### Copy to VM:

```bash
gcloud compute scp --recurse mongo_dumps watcher-mongo:~/ --zone=us-central1-a
```

#### Restore MongoDB on the VM SSH into VM:

- SSH into the VM:

```bash
gcloud compute ssh watcher-mongo --zone=us-central1-a
```

- Restore the collections:

```bash
mongorestore --drop --db watcher --collection tmdb_metadata tmdb_metadata.bson
mongorestore --drop --db watcher --collection watch_history watch_history.bson
... (repeat for other collections)
```

- Verify data:

 ```bash
mongosh
> use watcher
> db.tmdb_metadata.count()
 ```

## Build and Push Backend Docker Image

### Build image on GCP Cloud Build and push to Container Registry
In the directory containing the backend Dockerfile, run:
```bash
export BACKEND_IMAGE="us-central1-docker.pkg.dev/$PROJECT_ID/watcher/api"
export REGION=us-central1
export MONGO_VM_IP="<VM_EXTERNAL_IP>"  # replace with your mongo VMs external IP
gcloud builds submit . --tag $BACKEND_IMAGE
```

### Deploy Backend to Cloud Run
Deploy with all required environment variables:

**Note:** The backend requires 8Gi memory to accommodate:
- FAISS index loaded into memory
- Embedding model (sentence-transformers)
- Application overhead

```bash
gcloud run deploy watcher-backend \
--image $BACKEND_IMAGE \
--region $REGION \
--platform managed \
--allow-unauthenticated \
--memory 8Gi \
--cpu 2 \
--concurrency 1 \
--timeout 900 \
--set-env-vars \
MONGODB_URI="mongodb://$GCP_MONGO_DB_USER:$GCP_MONGO_DB_PASSWORD@$MONGO_VM_IP:27017/watcher?authSource=admin",\
EMBED_DEVICE=cpu,\
FAISS_USE_GPU=false,\
FAISS_SOURCE=$FAISS_SOURCE,\
FAISS_BUCKET=$FAISS_BUCKET,\
FAISS_PREFIX=$FAISS_PREFIX,\
OPENAI_API_KEY=$OPENAI_API_KEY,\
TMDB_API_KEY=$TMDB_API_KEY,\
TRAKT_CLIENT_ID=$TRAKT_CLIENT_ID,\
TRAKT_CLIENT_SECRET=$TRAKT_CLIENT_SECRET,\
TRAKT_REDIRECT_URI=$TRAKT_REDIRECT_URI,\
UI_BASE_URL=$UI_BASE_URL
```

The `?authSource=admin` parameter in the `MONGODB_URI` is necessary since we created the user in the `admin` database.
Once the backend is deployed, we replace `TRAKT_REDIRECT_URI` with the actual deployed URI (https://watcher-backend-391638080074.us-central1.run.app/auth/trakt/callback):
and the UI URL (https://watcher-ui-391638080074.us-central1.run.app) in `UI_BASE_URL` after deploying the UI (see steps below).

```bash
gcloud run services update watcher-backend \
--region $REGION \
--set-env-vars TRAKT_REDIRECT_URI=<ACTUAL_REDIRECT_URI>
```

For the FAISS setup, please refer to the FAISS section below. Use the required environment variables when deploying the backend.

## Build and Push UI Docker Image
In the `ui` directory which has the Dockerfile for the UI, run:

### Build image on GCP Cloud Build and push to Container Registry
```bash
export UI_IMAGE="us-central1-docker.pkg.dev/$PROJECT_ID/watcher/ui"
gcloud builds submit . --tag $UI_IMAGE
```

### Deploy UI to Cloud Run
Deploy with all required environment variables:

```bash
gcloud run deploy watcher-ui \
--image $UI_IMAGE \
--region us-central1 \
--platform managed \
--memory 512Mi \
--allow-unauthenticated \
--set-env-vars \
API_BASE_URL=<BACKEND_URL>,\
IMAGE_DIR="./static/images"
```

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
- Cloud Run → download FAISS from GCS at startup into `/tmp/faiss`
- **Memory requirement:** The backend needs at least 8Gi memory to load the FAISS index and embedding model into memory

The backend uses these environment variables:

```bash
FAISS_SOURCE=gcs  # or unset for local (defaults to ./faiss_index/ if unset)
FAISS_BUCKET=watcher-faiss
FAISS_PREFIX=v1  # versioned folder in bucket
```

### Local Embedding Computation
**Optional**: Run a TMDB sync to get any new items into the local MongoDB - FAISS embeddings are computed during the sync.

To recompute embeddings, start the UI using `./start.sh`, access the Admin panel, and perform a full embedding rebuild (FAISS tab).

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

#### Update the backend service (no rebuild required):
```bash
gcloud run services update watcher-backend \
--region $REGION \
--set-env-vars FAISS_SOURCE=gcs,FAISS_BUCKET=$FAISS_BUCKET,FAISS_PREFIX=v1
```

On the next cold start, the backend will:
- Download FAISS files from GCS
- Store them in /tmp/faiss
- Load the index into memory

#### Keeping FAISS Updated
Whenever you need to update the FAISS index or embeddings (e.g., after adding new items), repeat the upload step to overwrite the files in the GCS bucket.
```bash
gsutil -m cp \
tmdb.index \
labels.npy \
vecs.npy \
sidecar_meta.json \
gs://$FAISS_BUCKET/v2/
```

Then update the backend service to point to the new prefix:
```bash
gcloud run services update watcher-backend \
--region $REGION \
--set-env-vars FAISS_PREFIX=v2
```

This will trigger a restart of the backend service, which will load the updated FAISS index, with zero downtime and no new image creation required.

## Verify Deployment

The Watcher app should now be fully deployed! Access the UI URL provided by Cloud Run to start using the application.