# ============================================================
#  Sync Manager — Cross-device sync via Supabase
# ============================================================

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg


class SyncManager:
    def __init__(self, memory=None):
        """Initializes the sync manager, setting up the necessary configurations for background syncing."""
        self.memory = memory
        self.enabled = False
        self.connected = False
        self._client = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self.last_sync_at: Optional[str] = None
        self.last_error = ""
        self.push_count = 0
        self.pull_count = 0
        self.reconfigure()

    def _log(self, action: str, status: str, details: str = "", latency_ms: Optional[float] = None):
        """Logs sync actions and their status, including optional latency information."""
        if self.memory:
            try:
                self.memory.structured.log_sync_event("Supabase", action, status, details=details, latency_ms=latency_ms)
            except Exception:
                pass

    def _credentials(self) -> tuple[str, str]:
        """Retrieves the credentials needed for syncing (e.g., API keys)."""
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_KEY", "").strip()
        if self.memory:
            url = url or self.memory.structured.get_preference("supabase_url", cfg.SUPABASE_URL)
            key = key or self.memory.structured.get_preference("supabase_key", cfg.SUPABASE_KEY)
        else:
            url = url or cfg.SUPABASE_URL
            key = key or cfg.SUPABASE_KEY
        return url.strip(), key.strip()

    def reconfigure(self):
        """Reconfigures the sync manager with new settings or credentials."""
        with self._lock:
            self.enabled = False
            self.connected = False
            self._client = None
            url, key = self._credentials()
            if not (url and key):
                self.last_error = "Supabase credentials not configured."
                return
            try:
                from supabase import create_client
                start = time.perf_counter()
                self._client = create_client(url, key)
                self._client.table("aria_tasks").select("id").limit(1).execute()
                self.enabled = True
                self.connected = True
                self.last_error = ""
                self._log("connect", "ok", latency_ms=round((time.perf_counter() - start) * 1000, 2))
            except ImportError:
                self.last_error = "supabase package not installed."
                self._log("connect", "error", self.last_error)
            except Exception as e:
                self.last_error = str(e)
                self._log("connect", "error", self.last_error)

    def start_background_sync(self):
        """Starts the background synchronization process, ensuring data is kept in sync across devices."""
        if not self.enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the sync manager and halts all background syncing processes."""
        self._running = False

    def _sync_loop(self):
        """Main loop that handles the continuous synchronization of data."""
        while self._running:
            try:
                self.sync_now()
            except Exception:
                pass
            time.sleep(cfg.SYNC_INTERVAL_SECS)

    def sync_now(self) -> dict:
        """Performs an immediate sync and returns the result as a dictionary."""
        if not self.enabled or not self._client or not self.memory:
            return self.sync_status()
        started = time.perf_counter()
        try:
            self.push_all(); self.pull_remote()
            self.last_sync_at = datetime.now().isoformat()
            self.connected = True
            self.last_error = ""
            self._log("sync", "ok", latency_ms=round((time.perf_counter()-started)*1000, 2))
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            self._log("sync", "error", self.last_error, latency_ms=round((time.perf_counter()-started)*1000, 2))
        return self.sync_status()

    def push_all(self):
        """Pushes all locally stored data to the remote server for syncing."""
        if not self.enabled or not self._client or not self.memory:
            return
        for task in self.memory.structured.get_tasks():
            try:
                self._client.table("aria_tasks").upsert({**task, "device": cfg.DEVICE_NAME}).execute()
            except Exception:
                pass
        for key, value in self.memory.structured.get_all_facts().items():
            try:
                self._client.table("aria_facts").upsert({"key": key, "value": value, "device": cfg.DEVICE_NAME, "updated_at": datetime.now().isoformat()}).execute()
            except Exception:
                pass
        self.push_count += 1

    def pull_remote(self):
        """Pulls data from the remote server to sync with the local system."""
        if not self.enabled or not self._client or not self.memory:
            return
        try:
            resp = self._client.table("aria_tasks").select("*").neq("device", cfg.DEVICE_NAME).execute()
            for item in resp.data or []:
                self.memory.structured.upsert_task_record(item)
            resp2 = self._client.table("aria_facts").select("*").execute()
            for fact in resp2.data or []:
                self.memory.structured.set_fact(fact["key"], fact["value"])
        finally:
            self.pull_count += 1

    def sync_status(self) -> dict:
        """Returns the current sync status, including any pending operations or errors."""
        url, key = self._credentials()
        return {
            "enabled": self.enabled,
            "configured": bool(url and key),
            "connected": self.connected,
            "provider": "Supabase" if url else "local only",
            "device": cfg.DEVICE_NAME,
            "last_sync_at": self.last_sync_at,
            "last_error": self.last_error,
            "push_count": self.push_count,
            "pull_count": self.pull_count,
        }


SUPABASE_SETUP_SQL = """
CREATE TABLE IF NOT EXISTS aria_tasks (
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
    result TEXT,
    device TEXT
);
CREATE TABLE IF NOT EXISTS aria_facts (
    key TEXT PRIMARY KEY,
    value TEXT,
    device TEXT,
    updated_at TEXT
);
"""
