import redis

r = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=True,
)


def set_cached(key: str, value: str, ttl: int = 3600) -> None:
    r.setex(key, ttl, value)


def get_cached(key: str) -> str | None:
    result = r.get(key)
    return str(result) if result is not None else None
