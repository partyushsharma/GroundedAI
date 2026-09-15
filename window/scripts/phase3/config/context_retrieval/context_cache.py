import sqlite3
import hashlib
from pathlib import Path
from typing import Optional

CACHE_DB = Path("data/cache/context_cache.db")
CACHE_DB.parent.mkdir(parents=True, exist_ok=True)


class ContextCache:
    def __init__(self, db_path: Path = CACHE_DB):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS context_cache (
                key TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    @staticmethod
    def make_key(doc_id: str, chunk_index: int, chunk_text: str, prompt_version: str) -> str:
        """Stable hash of everything that affects the generated context."""
        payload = f"{doc_id}|{chunk_index}|{chunk_text}|{prompt_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT context FROM context_cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, context: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO context_cache (key, context) VALUES (?, ?)",
            (key, context),
        )
        self.conn.commit()

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM context_cache").fetchone()[0]
        return {"cached_entries": total}