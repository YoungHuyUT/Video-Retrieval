"""Two-tier cache manager (Improvement.md Task 9).

Provides a unified API wrapping:
- **In-process LRU** (fastest, lost on restart) for within-session repeats.
- **On-disk persistent cache** (`diskcache`) for expensive results that survive restarts.

Cache keys MUST include model version to auto-invalidate on model change.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/cache")


class CacheManager:
    """Two-tier cache: in-process LRU + on-disk persistent (diskcache)."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        disk_max_bytes: int = 500 * 1024 * 1024,  # 500 MB
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._disk: Any | None = None
        self._disk_max_bytes = disk_max_bytes
        self._lru: dict[str, Any] = {}
        self._lru_order: list[str] = []
        self._lru_maxsize = 2000
        self._init_disk()

    def _init_disk(self) -> None:
        """Initialize diskcache (lazy import, graceful fallback)."""
        try:
            import diskcache

            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._disk = diskcache.Cache(
                str(self._cache_dir),
                size_limit=self._disk_max_bytes,
            )
            logger.info("Disk cache initialized at %s", self._cache_dir)
        except ImportError:
            logger.warning(
                "diskcache not installed — on-disk cache disabled. "
                "Install with: pip install diskcache"
            )
            self._disk = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to init disk cache: %s", exc)
            self._disk = None

    @staticmethod
    def make_key(*parts: Any, **kwargs: Any) -> str:
        """Create a stable SHA-256 cache key from arbitrary parts and kwargs."""
        raw = json.dumps({"args": parts, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ── Disk (persistent) ──────────────────────────────────────────────

    def disk_get(self, key: str) -> Any | None:
        """Retrieve from disk cache. Returns None on miss."""
        if self._disk is None:
            return None
        try:
            return self._disk.get(key)
        except Exception:  # noqa: BLE001
            return None

    def disk_set(self, key: str, value: Any, tag: str | None = None) -> None:
        """Store in disk cache."""
        if self._disk is None:
            return
        try:
            self._disk.set(key, value, tag=tag)
        except Exception:  # noqa: BLE001
            logger.debug("Disk cache set failed for key %s", key[:12])

    def disk_clear(self, tag: str | None = None) -> int:
        """Clear disk cache entries. Returns count of removed entries."""
        if self._disk is None:
            return 0
        try:
            if tag:
                return self._disk.evict(tag)
            else:
                count = len(self._disk)
                self._disk.clear()
                return count
        except Exception:  # noqa: BLE001
            return 0

    # ── LRU (in-process) ───────────────────────────────────────────────

    def lru_get_or_set(self, key: str, factory: Any) -> Any:
        """Get from in-process LRU, or compute and store."""
        if key in self._lru:
            return self._lru[key]

        value = factory() if callable(factory) else factory
        self._lru[key] = value
        self._lru_order.append(key)

        # Evict oldest if over capacity
        while len(self._lru) > self._lru_maxsize:
            old_key = self._lru_order.pop(0)
            self._lru.pop(old_key, None)

        return value

    def lru_clear(self) -> None:
        """Clear in-process LRU cache."""
        self._lru.clear()
        self._lru_order.clear()

    # ── Convenience: combined get/set ───────────────────────────────────

    def get(self, key: str, persistent: bool = True) -> Any | None:
        """Get from cache (disk first, then LRU)."""
        if persistent:
            result = self.disk_get(key)
            if result is not None:
                return result
        return self._lru.get(key)

    def set(
        self,
        key: str,
        value: Any,
        persistent: bool = True,
        tag: str | None = None,
    ) -> None:
        """Store in both LRU and optionally disk."""
        if key not in self._lru:
            self._lru_order.append(key)
        self._lru[key] = value
        # Evict oldest if over capacity
        while len(self._lru) > self._lru_maxsize:
            old_key = self._lru_order.pop(0)
            self._lru.pop(old_key, None)
        if persistent:
            self.disk_set(key, value, tag=tag)

    def clear(self, tag: str | None = None) -> None:
        """Clear all caches."""
        self.lru_clear()
        self.disk_clear(tag=tag)
