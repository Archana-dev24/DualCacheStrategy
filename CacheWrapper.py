import hashlib
import json
import logging
import time
from dataclasses import dataclass

import redis
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

logger = logging.getLogger(__name__)

# Eviction policies that make sense for a cache workload
_VALID_POLICIES = {
    "volatile-lru",     # evict LRU keys that have a TTL  (default — safest)
    "volatile-lfu",     # evict LFU keys that have a TTL
    "volatile-ttl",     # evict keys closest to expiry first
    "volatile-random",  # evict random TTL keys
    "allkeys-lru",      # evict any LRU key regardless of TTL
    "allkeys-lfu",      # evict any LFU key regardless of TTL
    "allkeys-random",   # evict any random key
    "noeviction",       # return error on write when full (not recommended for cache)
}


def configure_eviction(
    client: redis.Redis,
    max_memory: str = "256mb",
    policy: str = "volatile-lru",
) -> dict:
    """
    Apply Redis maxmemory + eviction policy so the cache self-manages under memory pressure.

    When memory hits max_memory Redis will automatically evict entries according
    to the chosen policy before accepting new writes — no application logic needed.

    Args:
        client:     shared Redis connection
        max_memory: size string Redis understands, e.g. "256mb", "1gb", "0" (unlimited)
        policy:     one of volatile-lru (default), volatile-lfu, volatile-ttl,
                    allkeys-lru, allkeys-lfu, allkeys-random, noeviction

    Returns:
        dict with the active maxmemory and maxmemory-policy values confirmed from Redis
    """
    if policy not in _VALID_POLICIES:
        raise ValueError(f"Unknown eviction policy '{policy}'. Valid options: {sorted(_VALID_POLICIES)}")

    client.config_set("maxmemory", max_memory)
    client.config_set("maxmemory-policy", policy)

    active = client.config_get(["maxmemory", "maxmemory-policy"])
    logger.info(
        "Redis eviction configured — maxmemory=%s  policy=%s",
        active.get("maxmemory", "?"),
        active.get("maxmemory-policy", "?"),
    )
    return active


@dataclass
class CacheResult:
    response: str
    source: str       # "L1" | "L2"
    latency_ms: float


class L1HashCache:
    """
    Exact-match cache keyed on SHA-256 hash of the normalized query.
    O(1) Redis GET — fastest possible lookup.
    """

    PREFIX = "l1:"

    def __init__(self, client: redis.Redis, ttl: int = 3600):
        self.client = client
        self.ttl = ttl

    def _key(self, query: str) -> str:
        normalized = query.strip().lower()
        return self.PREFIX + hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        raw = self.client.get(self._key(query))
        return json.loads(raw)["response"] if raw else None

    def set(self, query: str, response: str) -> None:
        self.client.setex(
            self._key(query),
            self.ttl,
            json.dumps({"prompt": query, "response": response}),
        )

    def delete(self, query: str) -> bool:
        return self.client.delete(self._key(query)) > 0

    def flush(self) -> int:
        keys = self.client.keys(f"{self.PREFIX}*")
        return self.client.delete(*keys) if keys else 0

    def size(self) -> int:
        return len(self.client.keys(f"{self.PREFIX}*"))


class L2SemanticCache:
    """
    Approximate-match cache using RedisVL SemanticCache.
    Catches paraphrases and semantically similar queries within the
    configured cosine distance threshold.
    """

    def __init__(
        self,
        client: redis.Redis,
        ttl: int = 3600,
        distance_threshold: float = 0.15,
        index_name: str = "dual-l2-cache",
    ):
        vectorizer = HFTextVectorizer(
            model="redis/langcache-embed-v1",
            cache=EmbeddingsCache(redis_client=client, ttl=ttl),
        )
        self._cache = SemanticCache(
            name=index_name,
            vectorizer=vectorizer,
            redis_client=client,
            distance_threshold=distance_threshold,
        )
        self._cache.set_ttl(ttl)

    def get(self, query: str) -> str | None:
        results = self._cache.check(query)
        return results[0]["response"] if results else None

    def set(self, query: str, response: str) -> None:
        self._cache.store(prompt=query, response=response)

    def set_threshold(self, threshold: float) -> None:
        self._cache.distance_threshold = threshold


class DualCache:
    """
    Two-tier cache pipeline:

        Query → L1 (SHA-256 exact match) → L2 (semantic similarity) → LLM

    L1 is always checked first for near-zero-latency exact matches.
    On an L1 miss, L2 is searched for semantically similar responses.
    Any L2 hit is backfilled into L1 so the next identical query never
    reaches L2. The caller handles the LLM fallback and calls store()
    to write through to both layers.
    """

    def __init__(self, l1: L1HashCache, l2: L2SemanticCache):
        self.l1 = l1
        self.l2 = l2

    def check(self, query: str) -> CacheResult | None:
        t0 = time.perf_counter()

        # ── L1: exact match ──────────────────────────────────────
        response = self.l1.get(query)
        if response is not None:
            return CacheResult(
                response=response,
                source="L1",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── L2: semantic match ───────────────────────────────────
        response = self.l2.get(query)
        if response is not None:
            self.l1.set(query, response)   # backfill L1 for next time
            return CacheResult(
                response=response,
                source="L2",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return None   # full miss — caller queries LLM then calls store()

    def store(self, query: str, response: str) -> None:
        """Write-through to both L1 and L2."""
        self.l1.set(query, response)
        self.l2.set(query, response)

    def stats(self) -> dict:
        mem_info  = self.l1.client.info("memory")
        evict_cfg = self.l1.client.config_get(["maxmemory", "maxmemory-policy"])

        used_mb  = mem_info["used_memory"] / 1024 / 1024
        max_bytes = int(evict_cfg.get("maxmemory", 0))
        max_mb    = max_bytes / 1024 / 1024 if max_bytes > 0 else None
        used_pct  = (used_mb / max_mb * 100) if max_mb else None

        return {
            "l1_entries":       self.l1.size(),
            "l1_ttl_seconds":   self.l1.ttl,
            "l2_threshold":     self.l2._cache.distance_threshold,
            "memory_used_mb":   round(used_mb, 2),
            "memory_max_mb":    round(max_mb, 2) if max_mb else "unlimited",
            "memory_used_pct":  f"{used_pct:.1f}%" if used_pct is not None else "n/a",
            "eviction_policy":  evict_cfg.get("maxmemory-policy", "unknown"),
        }
