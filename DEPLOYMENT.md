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

```bash
gcloud run deploy watcher-backend \
--image $BACKEND_IMAGE \
--region $REGION \
--platform managed \
--allow-unauthenticated \
--memory 4Gi \
--cpu 2 \
--concurrency 1 \
--timeout 900 \
--set-env-vars \
MONGODB_URI="mongodb://$GCP_MONGO_DB_USER:$GCP_MONGO_DB_PASSWORD@$MONGO_VM_IP:27017/watcher?authSource=admin",\
EMBED_DEVICE=cpu,\
FAISS_USE_GPU=false,\
OPENAI_API_KEY=<OPENAI_API_KEY>,\
TMDB_API_KEY=<TMDB_API_KEY>,\
TRAKT_CLIENT_ID=<TRAKT_CLIENT_ID>,\
TRAKT_CLIENT_SECRET=<TRAKT_CLIENT_SECRET>,\
TRAKT_REDIRECT_URI="http://localhost:8080/auth/trakt/callback"
```

The `?authSource=admin` parameter in the `MONGODB_URI` is necessary since we created the user in the `admin` database.
Once the backend is deployed, we replace `TRAKT_REDIRECT_URI` with the actual deployed URI:

```bash
gcloud run services update watcher-backend \
--region $REGION \
--set-env-vars TRAKT_REDIRECT_URI=<ACTUAL_REDIRECT_URI>
```

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

## Verify Deployment

The Watcher app should now be fully deployed! Access the UI URL provided by Cloud Run to start using the application.

## Future Steps

- Add precomputed embeddings and FAISS index to the backend image or make them accessible to Cloud Run.
- Update the backend Dockerfile to include FAISS index if you want similarity search to work.
- Rebuild the FAISS index whenever new TMDB items are added:
  - Locally, run `./start.sh` to start the UI, use the Admin panel to perform a TMDB sync and rebuild the FAISS index.
  - Then, dump the MongoDB collections and copy them to the GCP VM as done earlier. This will update the data used by the backend on GCP.
  - To update the FAISS index in the backend container, either:
    - Rebuild and redeploy the backend Docker image with the updated FAISS index.
    - Or, modify the backend to load the FAISS index from a shared storage location accessible by Cloud Run (instead of having it as part of the Docker image).