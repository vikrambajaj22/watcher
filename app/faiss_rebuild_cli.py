"""CLI entrypoint to rebuild FAISS index in a separate process.

Usage:
  python -m app.faiss_rebuild_cli --dim 768 --factory "IDMap,IVF100,Flat"

This isolates the heavy training step from web server workers (prevents worker OOM/crashes).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from app.vector_store import rebuild_index
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index (isolated process)"
    )
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument(
        "--factory",
        type=str,
        default="IDMap,IVF100,Flat",
        help="FAISS index factory string",
    )
    args = parser.parse_args(argv)

    logger.info("Starting FAISS rebuild (dim=%s, factory=%s)", args.dim, args.factory)
    try:
        idx = rebuild_index(dim=args.dim, factory=args.factory)
        if idx is None:
            logger.info("Rebuild finished: no index created (no embeddings?)")
        else:
            logger.info("Rebuild finished: index created/loaded")
        return 0
    except Exception as e:
        logger.exception("FAISS rebuild failed: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
