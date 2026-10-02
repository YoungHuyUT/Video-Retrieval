"""Cache invalidation utility (Improvement.md Task 9).

Usage:
    python scripts/clear_cache.py --category llm       # Clear LLM analyzer cache
    python scripts/clear_cache.py --category blip2      # Clear BLIP-2 score cache
    python scripts/clear_cache.py --category object     # Clear object detection cache
    python scripts/clear_cache.py --category asr        # Clear ASR embedding cache
    python scripts/clear_cache.py --category all        # Clear everything
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear AIC2026 pipeline caches")
    parser.add_argument(
        "--category",
        choices=["llm", "blip2", "object", "asr", "verifier", "all"],
        required=True,
        help="Cache category to clear",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_CACHE_DIR),
        help="Cache directory (default: data/cache/)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.info("Cache directory does not exist: %s — nothing to clear", cache_dir)
        return

    if args.category == "all":
        _clear_all(cache_dir)
    else:
        _clear_by_tag(cache_dir, args.category)


def _clear_all(cache_dir: Path) -> None:
    """Remove entire cache directory."""
    import shutil

    count_before = sum(1 for _ in cache_dir.rglob("*"))
    shutil.rmtree(cache_dir, ignore_errors=True)
    logger.info("Cleared ALL cache (%d files removed) from %s", count_before, cache_dir)


def _clear_by_tag(cache_dir: Path, tag: str) -> None:
    """Clear entries with a specific tag from diskcache."""
    try:
        import diskcache

        cache = diskcache.Cache(str(cache_dir))
        count = cache.evict(tag)
        logger.info("Cleared %d entries with tag '%s' from %s", count, tag, cache_dir)
        cache.close()
    except ImportError:
        logger.warning("diskcache not installed — cannot clear by tag")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to clear cache: %s", exc)


if __name__ == "__main__":
    main()
