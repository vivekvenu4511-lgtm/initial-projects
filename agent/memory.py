# ============================================================
#  Memory System (Vector + Structured)
# ============================================================

from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

try:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except Exception:
    chromadb = None
    DefaultEmbeddingFunction = None

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


class VectorMemory:
    def __init__(self):
        """Initializes the memory class, setting up necessary data structures to store conversations and documents."""
        self._fallback = chromadb is None
        if not self._fallback:
            try:
                self.client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
                ef = DefaultEmbeddingFunction()
                self.conv = self.client.get_or_create_collection("conversations", embedding_function=ef)
                self.docs = self.client.get_or_create_collection("documents", embedding_function=ef)
            except Exception as e:
                print(f"  [Memory] ChromaDB init failed ({e}), using in-memory fallback.")
                self._fallback = True
        if self._fallback:
            self._conv_rows: list[dict] = []
            self._doc_rows: dict[str, dict] = {}

    def save_conversation(self, user_msg: str, agent_reply: str):
        """Saves a conversation between the user and the agent, including the user message and agent reply."""
        if self._fallback:
            self._conv_rows.append({"text": f"User: {user_msg}\nAgent: {agent_reply}", "ts": datetime.now().isoformat()})
            self._conv_rows = self._conv_rows[-500:]
            return
        self.conv.add(
            ids=[str(uuid.uuid4())],
            documents=[f"User: {user_msg}\nAgent: {agent_reply}"],
            metadatas=[{"ts": datetime.now().isoformat(), "device": cfg.DEVICE_NAME}],
        )

    def save_document(self, text: str, source: str, chunk_index: int = 0):
        """Saves a document to memory, storing its text, source, and an optional chunk index for large documents."""
        if self._fallback:
            self._doc_rows[f"{source}_{chunk_index}"] = {"text": text, "source": source, "chunk": chunk_index}
            return
        self.docs.upsert(
            ids=[f"{source}_{chunk_index}"],
            documents=[text],
            metadatas=[{"source": source, "chunk": chunk_index, "indexed_at": datetime.now().isoformat()}],
        )

    def search_conversations(self, query: str, n: int = cfg.MEMORY_RESULTS) -> list[str]:
        """Searches the saved conversations for a specific query, returning the most relevant results."""
        if self._fallback:
            q = (query or "").lower().strip()
            rows = list(reversed(self._conv_rows))
            if q:
                rows = sorted(rows, key=lambda r: ((q in r["text"].lower()), r["ts"]), reverse=True)
            return [r["text"] for r in rows[:n]]
        if self.conv.count() == 0:
            return []
        results = self.conv.query(query_texts=[query], n_results=min(n, self.conv.count()))
        return results.get("documents", [[]])[0] or []

    def search_documents(self, query: str, n: int = cfg.MEMORY_RESULTS) -> list[dict]:
        if self._fallback:
            q = (query or "").lower().strip()
            rows = list(self._doc_rows.values())
            if q:
                rows = sorted(rows, key=lambda r: ((q in r["text"].lower()) + (q in r["source"].lower())), reverse=True)
            return [{"text": r["text"], "source": r["source"]} for r in rows[:n]]
        if self.docs.count() == 0:
            return []
        results = self.docs.query(query_texts=[query], n_results=min(n, self.docs.count()))
        docs = results.get("documents", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []
        return [{"text": d, "source": (m or {}).get("source", "?")} for d, m in zip(docs, metas)]

    def get_sources(self) -> list[str]:
        if self._fallback:
            return sorted({r.get("source", "?") for r in self._doc_rows.values()})
        if self.docs.count() == 0:
            return []
        metas = self.docs.get(include=["metadatas"]).get("metadatas", [])
        return sorted({m.get("source", "?") for m in metas if m})

    def stats(self) -> dict:
        if self._fallback:
            return {"conversations": len(self._conv_rows), "document_chunks": len(self._doc_rows), "sources": len(self.get_sources()), "vector_backend": "fallback"}
        return {"conversations": self.conv.count(), "document_chunks": self.docs.count(), "sources": len(self.get_sources()), "vector_backend": "chromadb"}


class StructuredMemory:
    def __init__(self):
        self._init()

    def _conn(self):
        conn = sqlite3.connect(cfg.SQLITE_DB, check_same_thread=False, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str):
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _init(self):
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    priority TEXT DEFAULT 'normal',
                    scheduled_at TEXT,
                    reminder_minutes INTEGER DEFAULT 15,
                    reminder_sent_at TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    last_run_at TEXT,
                    run_count INTEGER DEFAULT 0,
                    result TEXT
                );
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT,
                    message TEXT,
                    data TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app TEXT,
                    window_title TEXT,
                    duration_secs INTEGER,
                    recorded_at TEXT
                );
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    filename TEXT,
                    filepath TEXT,
                    size INTEGER,
                    filetype TEXT,
                    uploaded_at TEXT,
                    indexed INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    filename TEXT,
                    filepath TEXT,
                    filetype TEXT,
                    size INTEGER,
                    source_prompt TEXT,
                    model_used TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS sync_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    action TEXT,
                    status TEXT,
                    latency_ms REAL,
                    details TEXT,
                    created_at TEXT
                );
                """
            )
            self._ensure_column(c, "tasks", "reminder_minutes", "INTEGER DEFAULT 15")
            self._ensure_column(c, "tasks", "reminder_sent_at", "TEXT")
            self._ensure_column(c, "tasks", "last_run_at", "TEXT")
            self._ensure_column(c, "tasks", "run_count", "INTEGER DEFAULT 0")
            self._ensure_column(c, "artifacts", "source_prompt", "TEXT")
            self._ensure_column(c, "artifacts", "model_used", "TEXT")

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def _one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(sql, params).fetchone()
        return dict(row) if row else None

    def add_task(self, title: str, description: str = "", priority: str = "normal", scheduled_at: Optional[str] = None, reminder_minutes: int = 15) -> str:
        tid = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO tasks (id,title,description,status,priority,scheduled_at,reminder_minutes,reminder_sent_at,created_at,updated_at,last_run_at,run_count,result)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (tid, title, description, "pending", priority, scheduled_at, int(reminder_minutes or 15), None, now, now, None, 0, None),
            )
        return tid

    def update_task(self, tid: str, status: Optional[str] = None, result: Optional[str] = None, **extra: Any):
        updates = dict(extra)
        if status is not None:
            updates["status"] = status
        if result is not None or "result" in extra:
            updates["result"] = result
        updates["updated_at"] = datetime.now().isoformat()
        if not updates:
            return
        cols = ", ".join(f"{k}=?" for k in updates)
        with self._conn() as c:
            c.execute(f"UPDATE tasks SET {cols} WHERE id=?", tuple(list(updates.values()) + [tid]))

    def delete_task(self, tid: str):
        with self._conn() as c:
            c.execute("DELETE FROM tasks WHERE id=?", (tid,))

    def get_task(self, tid: str) -> Optional[dict]:
        return self._one("SELECT * FROM tasks WHERE id=?", (tid,))

    def get_tasks(self, status: Optional[str] = None, limit: int = 200) -> list[dict]:
        order = "ORDER BY CASE WHEN status='pending' THEN 0 WHEN status='running' THEN 1 WHEN status='done' THEN 2 ELSE 3 END, CASE WHEN scheduled_at IS NULL OR scheduled_at='' THEN 1 ELSE 0 END, scheduled_at ASC, created_at DESC LIMIT ?"
        if status:
            return self._rows(f"SELECT * FROM tasks WHERE status=? {order}", (status, limit))
        return self._rows(f"SELECT * FROM tasks {order}", (limit,))

    def mark_task_run(self, tid: str, result: Optional[str] = None, status: str = "done"):
        task = self.get_task(tid)
        run_count = int((task or {}).get("run_count") or 0) + 1
        self.update_task(tid, status=status, result=result, last_run_at=datetime.now().isoformat(), run_count=run_count)

    def upsert_task_record(self, task: dict):
        now = datetime.now().isoformat()
        payload = {
            "id": task.get("id") or str(uuid.uuid4())[:8],
            "title": task.get("title", ""),
            "description": task.get("description", ""),
            "status": task.get("status", "pending"),
            "priority": task.get("priority", "normal"),
            "scheduled_at": task.get("scheduled_at"),
            "reminder_minutes": int(task.get("reminder_minutes") or 15),
            "reminder_sent_at": task.get("reminder_sent_at"),
            "created_at": task.get("created_at") or now,
            "updated_at": task.get("updated_at") or now,
            "last_run_at": task.get("last_run_at"),
            "run_count": int(task.get("run_count") or 0),
            "result": task.get("result"),
        }
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO tasks (id,title,description,status,priority,scheduled_at,reminder_minutes,reminder_sent_at,created_at,updated_at,last_run_at,run_count,result)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    payload["id"], payload["title"], payload["description"], payload["status"], payload["priority"], payload["scheduled_at"],
                    payload["reminder_minutes"], payload["reminder_sent_at"], payload["created_at"], payload["updated_at"], payload["last_run_at"], payload["run_count"], payload["result"],
                ),
            )

    def get_due_task_reminders(self, now: Optional[datetime] = None) -> list[dict]:
        now = now or datetime.now()
        due = []
        for task in self.get_tasks(status="pending", limit=500):
            if not task.get("scheduled_at") or task.get("reminder_sent_at"):
                continue
            try:
                due_at = datetime.fromisoformat(task["scheduled_at"])
            except Exception:
                continue
            remind_at = due_at - timedelta(minutes=int(task.get("reminder_minutes") or 15))
            if remind_at <= now <= due_at + timedelta(minutes=1):
                due.append(task)
        return due

    def mark_task_reminded(self, tid: str):
        self.update_task(tid, reminder_sent_at=datetime.now().isoformat())

    def set_fact(self, key: str, value: str):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO facts (key,value,updated_at) VALUES (?,?,?)", (key, value, datetime.now().isoformat()))

    def get_fact(self, key: str) -> Optional[str]:
        row = self._one("SELECT value FROM facts WHERE key=?", (key,))
        return row["value"] if row else None

    def get_all_facts(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT key,value FROM facts ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_preference(self, key: str, value: str):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO preferences (key,value,updated_at) VALUES (?,?,?)", (key, value, datetime.now().isoformat()))

    def get_preference(self, key: str, default: str = "") -> str:
        row = self._one("SELECT value FROM preferences WHERE key=?", (key,))
        return row["value"] if row and row.get("value") is not None else default

    def get_preferences(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT key,value FROM preferences ORDER BY key").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def log(self, level: str, message: str, data: Optional[dict] = None):
        with self._conn() as c:
            c.execute("INSERT INTO logs (level,message,data,created_at) VALUES (?,?,?,?)", (level, message, json.dumps(data, ensure_ascii=False) if data else None, datetime.now().isoformat()))

    def get_logs(self, limit: int = 80) -> list[dict]:
        return self._rows("SELECT * FROM logs ORDER BY created_at DESC LIMIT ?", (limit,))

    def log_activity(self, app: str, window: str, duration: int):
        with self._conn() as c:
            c.execute("INSERT INTO activity (app,window_title,duration_secs,recorded_at) VALUES (?,?,?,?)", (app, window, duration, datetime.now().isoformat()))

    def get_activity_summary(self, days: int = 7) -> list[dict]:
        return self._rows("SELECT app, SUM(duration_secs) as total_secs, COUNT(*) as sessions FROM activity WHERE recorded_at >= datetime('now', ?) GROUP BY app ORDER BY total_secs DESC LIMIT 15", (f"-{days} days",))

    def add_upload(self, uid: str, filename: str, filepath: str, size: int, filetype: str):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO uploads VALUES (?,?,?,?,?,?,?)", (uid, filename, filepath, size, filetype, datetime.now().isoformat(), 0))

    def get_uploads(self, limit: int = 100) -> list[dict]:
        return self._rows("SELECT * FROM uploads ORDER BY uploaded_at DESC LIMIT ?", (limit,))

    def mark_indexed(self, uid: str):
        with self._conn() as c:
            c.execute("UPDATE uploads SET indexed=1 WHERE id=?", (uid,))

    def add_artifact(self, filename: str, filepath: str, filetype: str, size: int, source_prompt: str = "", model_used: str = "") -> str:
        aid = str(uuid.uuid4())[:8]
        with self._conn() as c:
            c.execute("INSERT INTO artifacts (id,filename,filepath,filetype,size,source_prompt,model_used,created_at) VALUES (?,?,?,?,?,?,?,?)", (aid, filename, filepath, filetype, size, source_prompt, model_used, datetime.now().isoformat()))
        return aid

    def get_artifacts(self, limit: int = 100) -> list[dict]:
        return self._rows("SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?", (limit,))

    def log_sync_event(self, provider: str, action: str, status: str, details: str = "", latency_ms: Optional[float] = None):
        with self._conn() as c:
            c.execute("INSERT INTO sync_logs (provider,action,status,latency_ms,details,created_at) VALUES (?,?,?,?,?,?)", (provider, action, status, latency_ms, details, datetime.now().isoformat()))

    def get_sync_logs(self, limit: int = 80) -> list[dict]:
        return self._rows("SELECT * FROM sync_logs ORDER BY created_at DESC LIMIT ?", (limit,))


class AgentMemory:
    def __init__(self):
        self.vector = VectorMemory()
        self.structured = StructuredMemory()

    def remember(self, user_msg: str, agent_reply: str):
        self.vector.save_conversation(user_msg, agent_reply)

    def recall(self, query: str) -> str:
        parts = []
        convs = self.vector.search_conversations(query, 3)
        if convs:
            parts.append("=== Relevant past conversations ===\n" + "\n---\n".join(convs))
        docs = self.vector.search_documents(query, 4)
        if docs:
            parts.append("=== From your knowledge base ===\n" + "\n---\n".join(f"[{d['source']}]\n{d['text']}" for d in docs))
        facts = self.structured.get_all_facts()
        if facts:
            parts.append("=== What I know about you ===\n" + "\n".join(f"{k}: {v}" for k, v in facts.items()))
        return "\n\n".join(parts)

    def stats(self) -> dict:
        data = self.vector.stats()
        data["artifacts"] = len(self.structured.get_artifacts(1000))
        return data
