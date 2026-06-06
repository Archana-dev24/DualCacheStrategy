import logging
import os
import time

import redis
from dotenv import load_dotenv
from groq import Groq

from CacheWrapper import DualCache, L1HashCache, L2SemanticCache

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
L1_TTL         = int(os.getenv("L1_TTL", 3600))       # 1 hour  — exact matches
L2_TTL         = int(os.getenv("L2_TTL", 86400))      # 24 hours — semantic matches
L2_THRESHOLD   = float(os.getenv("L2_THRESHOLD", 0.15))
GROQ_MODEL     = "llama-3.3-70b-versatile"

# ── Shared Redis connection ─────────────────────────────────────────────────────
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
)

# ── Build dual cache ────────────────────────────────────────────────────────────
l1    = L1HashCache(client=redis_client, ttl=L1_TTL)
l2    = L2SemanticCache(client=redis_client, ttl=L2_TTL, distance_threshold=L2_THRESHOLD)
cache = DualCache(l1=l1, l2=l2)

# ── LLM client ──────────────────────────────────────────────────────────────────
groq_client = Groq()


def call_llm(question: str) -> str:
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant. "
                    "Answer concisely and professionally in 1-2 sentences."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_completion_tokens=256,
    )
    return completion.choices[0].message.content.strip()


def get_response(question: str) -> tuple[str, str, float]:
    """
    Full pipeline: L1 → L2 → LLM.

    Returns:
        (response, source, latency_ms)
        source is one of: "L1", "L2", "LLM"
    """
    t0 = time.perf_counter()

    result = cache.check(question)
    if result:
        logger.info("%-4s | %6.1f ms | %s", result.source, result.latency_ms, question[:70])
        return result.response, result.source, result.latency_ms

    # Full miss — call LLM and write through to both cache layers
    response = call_llm(question)
    cache.store(question, response)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("%-4s | %6.1f ms | %s", "LLM", latency_ms, question[:70])
    return response, "LLM", latency_ms


# ── Demo ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        # ── Round 1: cold cache — all LLM calls ──────────────────
        ("How do I reset my password?",           "LLM"),
        ("What are your business hours?",          "LLM"),
        ("How do I cancel my subscription?",       "LLM"),

        # ── Round 2: exact duplicates — L1 hits ──────────────────
        ("How do I reset my password?",            "L1"),
        ("What are your business hours?",           "L1"),

        # ── Round 3: paraphrases — L2 semantic hits ──────────────
        ("I forgot my password, how can I recover it?",  "L2"),
        ("When does your support team work?",             "L2"),
        ("I want to unsubscribe from my plan",            "L2"),

        # ── Round 4: L2 hit → backfilled → now L1 hit ────────────
        ("I forgot my password, how can I recover it?",  "L1"),
    ]

    SOURCE_ICON = {"L1": "✅ L1 ", "L2": "🔍 L2 ", "LLM": "🤖 LLM"}

    print("\n" + "═" * 72)
    print(f"  {'#':<3}  {'SOURCE':<7}  {'LATENCY':>9}  QUESTION")
    print("═" * 72)

    hits = {"L1": 0, "L2": 0, "LLM": 0}

    for i, (query, _expected) in enumerate(test_queries, 1):
        response, source, latency = get_response(query)
        hits[source] += 1
        icon = SOURCE_ICON.get(source, source)
        print(f"  {i:<3}  {icon:<9}  {latency:>7.1f} ms  {query}")

    total = len(test_queries)
    print("═" * 72)
    print(f"\n  Results: {hits['L1']} L1 hits  |  {hits['L2']} L2 hits  |  {hits['LLM']} LLM calls  ({total} total)")
    print(f"  Cache coverage: {((hits['L1'] + hits['L2']) / total * 100):.0f}%")

    stats = cache.stats()
    print(f"\n  Cache stats: {stats}")
    print()
