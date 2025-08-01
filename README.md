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
  --set-env-vars TRAKT_CLIENT_ID=xxx,TRAKT_CLIENT_SECRET=xxx,TRAKT_REDIRECT_URI=xxx,TMDB_API_KEY=xxx,TMDB_READ_ACCESS_TOKEN=xxx,MONGODB_URI=xxx,MONGODB_DB_NAME=watcher,OPENAI_CLIENT_ID=xxx,OPENAI_API_KEY=xxx
```

## Environment Setup
Create a `.env` file:
```env
TRAKT_CLIENT_ID=your-client-id
TRAKT_CLIENT_SECRET=your-client-secret
TRAKT_REDIRECT_URI=http://localhost:8080/auth/trakt/callback
TMDB_API_KEY=your-tmdb-key
TMDB_READ_ACCESS_TOKEN=your-tmdb-read-access-token
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=watcher
OPENAI_CLIENT_ID=your-openai-client-id
OPENAI_API_KEY=your-openai-secret-key
```

## Logging
Custom logger ensures clear logs for debugging and observability.

## License
MIT
