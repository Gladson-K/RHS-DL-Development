"""MongoDB + Redis access with in-memory fallbacks for offline tests."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - exercised only when the optional driver is present
    from pymongo import ASCENDING, MongoClient
except Exception:  # noqa: BLE001
    MongoClient = None  # type: ignore[assignment]
    ASCENDING = 1

try:  # pragma: no cover
    from redis import Redis
except Exception:  # noqa: BLE001
    Redis = None  # type: ignore[assignment]


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class MemoryCollection:
    """Tiny document store used when MongoDB is unavailable."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []

    def insert_one(self, doc: dict[str, Any]) -> None:
        self.docs.append(doc)

    def find(self, query: dict[str, Any] | None = None, limit: int = 100):
        query = query or {}

        def matches(doc: dict[str, Any]) -> bool:
            return all(doc.get(key) == value for key, value in query.items())

        return [doc for doc in self.docs if matches(doc)][:limit]

    def find_one(self, query: dict[str, Any]):
        rows = self.find(query, limit=1)
        return rows[0] if rows else None

    def count(self) -> int:
        return len(self.docs)


class MemoryDB:
    def __init__(self) -> None:
        self.collections: dict[str, MemoryCollection] = defaultdict(MemoryCollection)

    def __getitem__(self, name: str) -> MemoryCollection:
        return self.collections[name]

    def __getattr__(self, name: str) -> MemoryCollection:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.collections[name]


class MemoryRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.streams: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)

    def set(self, key: str, value: str) -> None:
        self.kv[key] = value

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def incr(self, key: str) -> int:
        self.counters[key] += 1
        return self.counters[key]

    def xadd(self, key: str, fields: dict[str, Any]) -> str:
        entry = {"id": f"{len(self.streams[key]) + 1}-0", "fields": fields}
        self.streams[key].append(entry)
        return entry["id"]

    def xrange(self, key: str, count: int = 200) -> list[dict[str, Any]]:
        return self.streams[key][-count:]

    def ping(self) -> bool:
        return True


class Store:
    """Thin facade over MongoDB + Redis with automatic fallback."""

    def __init__(self, mongo_uri: str, mongo_db: str, redis_url: str) -> None:
        self.mongo_uri = mongo_uri
        self.mongo_db_name = mongo_db
        self.redis_url = redis_url
        self.client = None
        self.db: Any = MemoryDB()
        self.redis: Any = MemoryRedis()
        self.mongo_ok = False
        self.redis_ok = False

    def connect(self) -> None:
        if MongoClient is not None:
            try:
                client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=3000)
                client.admin.command("ping")
                self.client = client
                self.db = client[self.mongo_db_name]
                self.mongo_ok = True
            except Exception:  # noqa: BLE001
                self.db = MemoryDB()
                self.mongo_ok = False
        if Redis is not None:
            try:
                redis = Redis.from_url(self.redis_url, decode_responses=True)
                redis.ping()
                self.redis = redis
                self.redis_ok = True
            except Exception:  # noqa: BLE001
                self.redis = MemoryRedis()
                self.redis_ok = False
        self.ensure_indexes()

    def ensure_indexes(self) -> None:
        if not self.mongo_ok:
            return
        try:
            self.db.artifacts.create_index([("run", ASCENDING), ("name", ASCENDING)])
            self.db.samples.create_index([("sample_id", ASCENDING)])
            self.db.validation_reports.create_index([("created_at", ASCENDING)])
        except Exception:  # noqa: BLE001, S110
            pass

    # ------------------------------------------------------------------ mongo
    def insert(self, collection: str, doc: dict[str, Any]) -> dict[str, Any]:
        doc = dict(doc)
        doc.setdefault("created_at", utcnow())
        self.db[collection].insert_one(doc)
        return doc

    def upsert(self, collection: str, key: str, value: Any, doc: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace one document keyed by ``key == value``."""
        doc = dict(doc)
        doc.setdefault("created_at", utcnow())
        try:
            if self.mongo_ok:
                self.db[collection].replace_one({key: value}, doc, upsert=True)
            else:
                mem = self.db[collection]
                mem.docs = [item for item in mem.docs if item.get(key) != value]
                mem.insert_one(doc)
        except Exception:  # noqa: BLE001, S110
            pass
        return doc

    def find(
        self,
        collection: str,
        query: dict[str, Any] | None = None,
        limit: int = 100,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        try:
            cursor = self.db[collection].find(query or {})
            if self.mongo_ok:
                cursor = cursor.limit(limit)
                if newest_first:
                    cursor = cursor.sort("created_at", -1)
                return [_clean(doc) for doc in cursor]
            rows = cursor
            if newest_first:
                rows = list(reversed(rows))
            return [_clean(doc) for doc in rows[:limit]]
        except Exception:  # noqa: BLE001
            return []

    def find_one(self, collection: str, key: str, value: Any) -> dict[str, Any] | None:
        doc = self.db[collection].find_one({key: value})
        return _clean(doc) if doc else None

    def count(self, collection: str) -> int:
        try:
            if self.mongo_ok:
                return int(self.db[collection].estimated_document_count())
            return int(self.db[collection].count())
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------ redis
    def set_json(self, key: str, value: Any) -> None:
        try:
            self.redis.set(key, json.dumps(value, default=str))
        except Exception:  # noqa: BLE001, S110
            pass

    def get_json(self, key: str) -> Any:
        try:
            raw = self.redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:  # noqa: BLE001
            return None

    def incr(self, key: str) -> int:
        try:
            return int(self.redis.incr(key))
        except Exception:  # noqa: BLE001
            return 0

    def xadd(self, stream: str, fields: dict[str, Any]) -> None:
        try:
            self.redis.xadd(stream, {key: str(value) for key, value in fields.items()})
        except Exception:  # noqa: BLE001
            try:
                self.redis.xadd(stream, fields)
            except Exception:  # noqa: BLE001, S110
                pass

    def xrange(self, stream: str, count: int = 200) -> list[dict[str, Any]]:
        try:
            rows = self.redis.xrange(stream, count=count)
            return [
                {
                    "id": row[0],
                    "fields": row[1] if isinstance(row, (tuple, list)) else row.get("fields", {}),
                }
                for row in rows
            ]
        except Exception:  # noqa: BLE001
            return []

    def status(self) -> dict[str, bool]:
        redis_ok = self.redis_ok
        try:
            redis_ok = bool(self.redis.ping())
        except Exception:  # noqa: BLE001
            redis_ok = False
        mongo_ok = self.mongo_ok
        if self.client is not None:
            try:
                self.client.admin.command("ping")
                mongo_ok = True
            except Exception:  # noqa: BLE001
                mongo_ok = False
        return {"mongo": mongo_ok, "redis": redis_ok}


def _clean(doc: dict[str, Any]) -> dict[str, Any]:
    if "_id" in doc:
        doc = dict(doc)
        doc["_id"] = str(doc["_id"])
    return doc
