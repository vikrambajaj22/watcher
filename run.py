import os

import uvicorn

if __name__ == "__main__":
    workers = int(os.getenv("UVICORN_WORKERS", "1"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, workers=workers)
