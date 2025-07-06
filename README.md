# watcher

> watchu lookin at?

watcher is a personal movie/TV recommendation app powered by Trakt, TMDB, and OpenAI LLMs that power poster and plot understanding.

## Features
- Syncs your Trakt history automatically
- Recommends content based on your watch patterns
- LLM-powered analysis of scripts or posters

## To Run Locally
```bash
uvicorn app.main:app --reload
```

## To Deploy on GCP Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/watcher

gcloud run deploy watcher \
  --image gcr.io/PROJECT_ID/watcher \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars TRAKT_ACCESS_TOKEN=xxx,TRAKT_CLIENT_ID=xxx,TMDB_API_KEY=xxx,MONGODB_URI=xxx,MONGODB_DB_NAME=watcher
```

## Environment Setup
Create a `.env` file:
```env
TRAKT_ACCESS_TOKEN=your-token
TRAKT_CLIENT_ID=your-client-id
TMDB_API_KEY=your-tmdb-key
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=watcher
```

## Logging
Custom logger ensures clear logs for debugging and observability.

## License
MIT
