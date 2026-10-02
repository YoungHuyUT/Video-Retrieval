"""Tests for CacheManager (Improvement.md Task 9)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root for scripts imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from aic2026.cache.cache_manager import CacheManager


class TestMakeKey:
    def test_deterministic(self):
        k1 = CacheManager.make_key("query", "v1")
        k2 = CacheManager.make_key("query", "v1")
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = CacheManager.make_key("query A")
        k2 = CacheManager.make_key("query B")
        assert k1 != k2

    def test_model_version_invalidation(self):
        k1 = CacheManager.make_key("query", model_version="qwen2.5:1.5b_v1")
        k2 = CacheManager.make_key("query", model_version="qwen2.5:1.5b_v2")
        assert k1 != k2

    def test_returns_hex_string(self):
        key = CacheManager.make_key("test")
        assert len(key) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in key)


class TestLRU:
    def test_hit_returns_stored_value(self):
        cm = CacheManager.__new__(CacheManager)
        cm._lru = {}
        cm._lru_order = []
        cm._lru_maxsize = 100
        cm._disk = None
        cm._cache_dir = Path("/tmp/test_cache")
        cm._disk_max_bytes = 0

        cm.set("key1", "value1", persistent=False)
        assert cm.get("key1", persistent=False) == "value1"

    def test_miss_returns_none(self):
        cm = CacheManager.__new__(CacheManager)
        cm._lru = {}
        cm._lru_order = []
        cm._lru_maxsize = 100
        cm._disk = None
        cm._cache_dir = Path("/tmp/test_cache")
        cm._disk_max_bytes = 0

        assert cm.get("nonexistent", persistent=False) is None

    def test_lru_eviction(self):
        cm = CacheManager.__new__(CacheManager)
        cm._lru = {}
        cm._lru_order = []
        cm._lru_maxsize = 3
        cm._disk = None
        cm._cache_dir = Path("/tmp/test_cache")
        cm._disk_max_bytes = 0

        cm.set("a", 1, persistent=False)
        cm.set("b", 2, persistent=False)
        cm.set("c", 3, persistent=False)
        cm.set("d", 4, persistent=False)  # evicts "a"

        assert cm.get("a", persistent=False) is None
        assert cm.get("b", persistent=False) == 2

    def test_lru_clear(self):
        cm = CacheManager.__new__(CacheManager)
        cm._lru = {}
        cm._lru_order = []
        cm._lru_maxsize = 100
        cm._disk = None
        cm._cache_dir = Path("/tmp/test_cache")
        cm._disk_max_bytes = 0

        cm.set("key1", "value1", persistent=False)
        cm.lru_clear()
        assert cm.get("key1", persistent=False) is None


class TestDiskCache:
    @pytest.fixture(autouse=True)
    def _require_diskcache(self):
        pytest.importorskip("diskcache")

    def test_disk_roundtrip(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path, disk_max_bytes=10_000_000)
        key = CacheManager.make_key("test_query")
        cm.disk_set(key, {"result": "hello"})
        assert cm.disk_get(key) == {"result": "hello"}

    def test_disk_miss_returns_none(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path, disk_max_bytes=10_000_000)
        assert cm.disk_get("nonexistent_key") is None

    def test_disk_clear_by_tag(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path, disk_max_bytes=10_000_000)
        cm.disk_set("k1", "v1", tag="llm")
        cm.disk_set("k2", "v2", tag="blip2")
        cleared = cm.disk_clear(tag="llm")
        assert cleared == 1
        assert cm.disk_get("k1") is None
        assert cm.disk_get("k2") == "v2"

    def test_disk_clear_all(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path, disk_max_bytes=10_000_000)
        cm.disk_set("k1", "v1")
        cm.disk_set("k2", "v2")
        cleared = cm.disk_clear()
        assert cleared == 2
        assert cm.disk_get("k1") is None


class TestCombinedCache:
    def test_persistent_get_checks_disk(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path, disk_max_bytes=10_000_000)
        key = CacheManager.make_key("query")
        cm.set(key, "hello", persistent=True)
        # Should find it via disk
        assert cm.get(key, persistent=True) == "hello"

    def test_clear_all(self, tmp_path):
        cm = CacheManager(cache_dir=tmp_path, disk_max_bytes=10_000_000)
        cm.set("k1", "v1", persistent=True)
        cm.clear()
        assert cm.get("k1", persistent=False) is None


class TestGracefulFallback:
    def test_no_diskcache_installed(self, tmp_path, monkeypatch):
        """CacheManager should not crash when diskcache is missing."""
        import importlib
        monkeypatch.setitem(sys.modules, "diskcache", None)
        # Re-import to trigger the import error path
        cm = CacheManager(cache_dir=tmp_path)
        assert cm._disk is None
        # Operations should be no-ops, not crashes
        cm.disk_set("key", "value")
        assert cm.disk_get("key") is None
