# DualCacheStrategy

A production-ready two-tier LLM response cache that combines **exact-match hashing (L1)** and **semantic similarity search (L2)** backed by Redis. Most repeated or paraphrased questions are served in milliseconds — the LLM is only called on a true cache miss.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  L1 — SHA-256 Hash Cache        │  ← O(1) Redis GET, sub-millisecond
│  Normalized exact match         │
└────────────┬────────────────────┘
             │ MISS
             ▼
┌─────────────────────────────────┐
│  L2 — Semantic Cache (RedisVL)  │  ← Vector similarity search
│  redis/langcache-embed-v1       │    catches paraphrases
└────────────┬────────────────────┘
             │ MISS                  L2 HIT → backfill L1
             ▼
┌─────────────────────────────────┐
│  Groq LLM                       │  ← llama-3.3-70b-versatile
│  llama-3.3-70b-versatile        │    write-through to L1 + L2
└─────────────────────────────────┘
```

**L1 backfill:** Every L2 hit is written back into L1, so the next identical query never reaches vector search.

---

## Project Structure

| File | Purpose |
|------|---------|
| `CacheWrapper.py` | `L1HashCache`, `L2SemanticCache`, `DualCache` — all cache logic |
| `main.py` | LLM client, `get_response()` pipeline, demo runner |
| `requirements.txt` | Python dependencies |

---

## Prerequisites

- Python 3.10+
- Redis Stack running locally on `localhost:6379`
  ```bash
  docker run -d -p 6379:6379 redis/redis-stack
  ```
- A Groq API key

---

## Installation

```bash
git clone https://github.com/Archana-dev24/DualCacheStrategy.git
cd DualCacheStrategy
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides (defaults shown)
REDIS_HOST=localhost
REDIS_PORT=6379
L1_TTL=3600       # L1 expiry in seconds (1 hour)
L2_TTL=86400      # L2 expiry in seconds (24 hours)
L2_THRESHOLD=0.15 # cosine distance cutoff for semantic match
```

---

## Usage

### Run the demo

```bash
python main.py
```

Sample output:

```
════════════════════════════════════════════════════════════════════════
  #    SOURCE     LATENCY  QUESTION
════════════════════════════════════════════════════════════════════════
  1    🤖 LLM     823.4 ms  How do I reset my password?
  2    🤖 LLM     741.2 ms  What are your business hours?
  3    🤖 LLM     698.5 ms  How do I cancel my subscription?
  4    ✅ L1        1.2 ms  How do I reset my password?
  5    ✅ L1        0.9 ms  What are your business hours?
  6    🔍 L2       38.7 ms  I forgot my password, how can I recover it?
  7    🔍 L2       41.2 ms  When does your support team work?
  8    🔍 L2       39.8 ms  I want to unsubscribe from my plan
  9    ✅ L1        0.8 ms  I forgot my password, how can I recover it?
════════════════════════════════════════════════════════════════════════

  Results: 4 L1 hits  |  3 L2 hits  |  3 LLM calls  (9 total)
  Cache coverage: 78%

  Cache stats: {'l1_entries': 6, 'l1_ttl_seconds': 3600, 'l2_threshold': 0.15}
```

Query 9 shows the **L1 backfill** in action — the paraphrase from query 6 was promoted to L1 and now returns in under 1 ms.

### Use as a module

```python
from main import get_response

response, source, latency_ms = get_response("How do I reset my password?")
# source → "L1", "L2", or "LLM"
# latency_ms → float
```

### Use the cache directly

```python
import redis
from CacheWrapper import DualCache, L1HashCache, L2SemanticCache

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

l1 = L1HashCache(client=r, ttl=3600)
l2 = L2SemanticCache(client=r, ttl=86400, distance_threshold=0.15)
cache = DualCache(l1=l1, l2=l2)

# Check both layers
result = cache.check("How do I reset my password?")
if result:
    print(result.response)   # CacheResult(response, source, latency_ms)

# Write-through to both layers
cache.store("How do I reset my password?", "Visit the login page and click Forgot Password.")

# Stats
print(cache.stats())
# {'l1_entries': 1, 'l1_ttl_seconds': 3600, 'l2_threshold': 0.15}
```

---

## CacheWrapper API

### `L1HashCache`

| Method | Description |
|--------|-------------|
| `get(query)` | Returns cached response string or `None` |
| `set(query, response)` | Stores entry with TTL |
| `delete(query)` | Removes a specific entry |
| `flush()` | Clears all `l1:*` keys |
| `size()` | Returns number of entries in L1 |

### `L2SemanticCache`

| Method | Description |
|--------|-------------|
| `get(query)` | Returns semantically similar response or `None` |
| `set(query, response)` | Stores entry in the vector index |
| `set_threshold(value)` | Adjust cosine distance threshold at runtime |

### `DualCache`

| Method | Description |
|--------|-------------|
| `check(query)` | Returns `CacheResult(response, source, latency_ms)` or `None` |
| `store(query, response)` | Write-through to both L1 and L2 |
| `stats()` | Returns L1 entry count, TTL, and L2 threshold |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `L1_TTL` | `3600` | L1 entry TTL in seconds |
| `L2_TTL` | `86400` | L2 entry TTL in seconds |
| `L2_THRESHOLD` | `0.15` | Cosine distance cutoff — lower = stricter |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |

**Tuning `L2_THRESHOLD`:**
- Too many wrong answers returned → lower the threshold
- Too many paraphrases falling through to LLM → raise the threshold

---

## How Each Layer Performs

| | L1 Hash | L2 Semantic | LLM |
|---|---|---|---|
| Match type | Exact (case/whitespace insensitive) | Approximate (cosine similarity) | N/A |
| Latency | ~1 ms | ~30–50 ms | ~700–1000 ms |
| Handles paraphrases | No | Yes | Yes |
| Redis operation | `GET` | Vector search | — |
| TTL | 1 hour (default) | 24 hours (default) | — |

---

## Tech Stack

- [Redis Stack](https://redis.io/docs/stack/) — key-value store + vector index
- [RedisVL](https://github.com/redis/redis-vl-python) — SemanticCache and vector search abstractions
- [Groq](https://groq.com/) — fast LLM inference (Llama 3.3 70B)
- [python-dotenv](https://pypi.org/project/python-dotenv/) — environment config
