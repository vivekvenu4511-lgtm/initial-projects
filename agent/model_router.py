# ============================================================
#  Model Router v7 — HuggingFace + llama-cpp + OpenRouter
#  Local backend: direct llama-cpp-python (no Ollama needed)
#  Ollama kept as optional fallback if running
# ============================================================

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg

logger = logging.getLogger("aria.model_router")

# Lazy-import local model manager to avoid heavy GPU init at import time
_local_mgr = None
_local_mgr_lock = None

def _get_local_manager():
    """Return the singleton LocalModelManager, initialised once."""
    global _local_mgr, _local_mgr_lock
    import threading
    if _local_mgr_lock is None:
        _local_mgr_lock = threading.Lock()
    with _local_mgr_lock:
        if _local_mgr is None:
            try:
                from agent.local_model_manager import LocalModelManager
                _local_mgr = LocalModelManager()
                logger.info("LocalModelManager (llama-cpp) initialised.")
            except Exception as e:
                logger.warning(f"LocalModelManager unavailable: {e}")
                _local_mgr = None
    return _local_mgr


def _trim_slash(value: str) -> str:
    return (value or "").rstrip("/")

def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)

def _safe_json_loads(value: str, default: Any):
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


@dataclass
class ChatResult:
    provider:   str
    model:      str
    content:    str
    latency_ms: float
    raw:        Optional[dict] = None


class ModelRouter:
    def __init__(self):
        self._cache: dict[str, tuple[float, Any]] = {}

    def _get_pref(self, key: str, default: str = "") -> str:
        try:
            with sqlite3.connect(cfg.SQLITE_DB) as c:
                row = c.execute("SELECT value FROM preferences WHERE key=?", (key,)).fetchone()
            if row and row[0] is not None:
                return str(row[0])
        except Exception:
            pass
        return default

    def _secret(self, env_key: str, pref_key: str, default: str = "") -> str:
        value = os.getenv(env_key, "").strip()
        return value or self._get_pref(pref_key, default).strip()

    def execution_mode(self) -> str:
        mode = (self._get_pref("execution_mode", cfg.DEFAULT_EXECUTION_MODE) or "").strip().lower()
        return mode if mode in {"local","cloud","local+cloud"} else cfg.DEFAULT_EXECUTION_MODE

    def openrouter_base(self) -> str:
        return _trim_slash(self._get_pref("openrouter_base", cfg.OPENROUTER_BASE) or cfg.OPENROUTER_BASE)

    def openrouter_key(self) -> str:
        return self._secret("OPENROUTER_API_KEY","openrouter_key",cfg.OPENROUTER_API_KEY)

    def refresh(self):
        self._cache.clear()  # clears catalog, ollama_status, openrouter_status caches

    def _cached(self, key: str):
        cached = self._cache.get(key)
        if not cached:
            return None
        ts, value = cached
        if time.time() - ts > cfg.MODEL_STATUS_CACHE_SECS:
            return None
        return value

    def _store(self, key: str, value: Any):
        self._cache[key] = (time.time(), value)
        return value

    # ─── custom cloud models ──────────────────────────────────

    def custom_cloud_models(self) -> list[dict]:
        raw   = self._get_pref("custom_cloud_models","[]")
        items = _safe_json_loads(raw,[])
        clean = []
        for item in items:
            if not isinstance(item,dict): continue
            model_id = (item.get("model") or item.get("id") or "").strip()
            if not model_id: continue
            clean.append({
                "uid":         item.get("uid") or str(uuid.uuid4())[:8],
                "id":          model_id,
                "name":        item.get("name") or model_id,
                "provider":    item.get("provider") or "custom",
                "base_url":    _trim_slash(item.get("base_url") or cfg.OPENROUTER_BASE),
                "api_key":     item.get("api_key") or "",
                "description": item.get("description") or "",
                "free":        bool(item.get("free",True)),
                "custom":      True,
            })
        return clean

    def save_custom_cloud_models(self, models: list[dict]):
        with sqlite3.connect(cfg.SQLITE_DB) as c:
            c.execute("INSERT OR REPLACE INTO preferences (key,value,updated_at) VALUES (?,?,datetime('now'))",
                      ("custom_cloud_models", json.dumps(models, ensure_ascii=False)))

    def add_custom_cloud_model(self, name, model, base_url, api_key="", description="", free=True):
        models  = [m for m in self.custom_cloud_models() if m.get("id") != model.strip()]
        uid_val = str(uuid.uuid4())[:8]
        entry   = {
            "uid":uid_val,"id":model.strip(),"name":(name or model).strip(),
            "provider":"custom","base_url":_trim_slash(base_url or cfg.OPENROUTER_BASE),
            "api_key":api_key.strip(),"description":description.strip(),
            "free":bool(free),"custom":True,
        }
        models.append(entry)
        self.save_custom_cloud_models(models)
        self.refresh()
        return entry

    def remove_custom_cloud_model(self, uid: str) -> bool:
        models = self.custom_cloud_models()
        new_models = [m for m in models if m.get("uid") != uid and m.get("id") != uid]
        changed = len(new_models) != len(models)
        if changed:
            self.save_custom_cloud_models(new_models)
            self.refresh()
        return changed

    # ─── task scoring ─────────────────────────────────────────

    def score_task(self, message: str, attachments: list[str] | None = None) -> dict:
        attachments = attachments or []
        msg   = (message or "").lower()
        score = 0
        task_type = "general"
        if len(message) > 200: score += 1
        if len(message) > 500: score += 2
        if len(attachments) == 1: score += 2
        if len(attachments) >= 2: score += 3
        code_kw  = ["code","python","javascript","debug","function","class","formula","vlookup","pivot","macro","script","regex","html","css","sql"]
        analysis_kw = ["analyze","analyse","compare","comparison","summary","report","chart","trend","pattern","statistics","insight","forecast","summarize"]
        gen_kw   = ["generate","create","write report","presentation","ppt","powerpoint","draft","compose","excel","spreadsheet","word document","docx","txt","markdown"]
        reason_kw = ["reason","think","solve","plan","strategy","advise","recommend","pros and cons","evaluate","decision"]
        is_code     = any(w in msg for w in code_kw)
        is_analysis = any(w in msg for w in analysis_kw)
        is_gen      = any(w in msg for w in gen_kw)
        is_reason   = any(w in msg for w in reason_kw)
        if is_code:     score += 2
        if is_analysis: score += 2
        if is_gen:      score += 2
        if is_reason:   score += 3
        # Priority: reasoning > coding > analysis > creative > general
        if is_reason:        task_type = "reasoning"
        elif is_code:        task_type = "coding"
        elif is_analysis:    task_type = "analysis"
        elif is_gen:         task_type = "creative"
        return {"score": score, "type": task_type}

    # ─── intelligence model selection ────────────────────────

    def intelligence_select(self, message: str, attachments: list[str] | None = None) -> dict:
        """Analyse prompt and pick the best available local+cloud models."""
        task = self.score_task(message, attachments)
        catalog = self.get_catalog(refresh=False)
        local_ids  = {m.get("id") for m in catalog.get("local_models",[])}
        cloud_list = catalog.get("cloud_models",[])
        cloud_ids  = {m.get("id") for m in cloud_list}

        def best_cloud(*candidates):
            for c in candidates:
                if c in cloud_ids: return c
            return catalog.get("selected_cloud_model")

        rationale = []
        t = task["type"]

        if t == "coding":
            chosen_cloud = best_cloud(cfg.CLOUD_MODEL_CODING, cfg.CLOUD_MODEL_BALANCED)
            chosen_local = cfg.OLLAMA_MODEL_CODE if cfg.OLLAMA_MODEL_CODE in local_ids else None
            rationale.append("Code task detected → selecting code-optimised models")
        elif t == "reasoning":
            chosen_cloud = best_cloud(cfg.CLOUD_MODEL_REASON, cfg.CLOUD_MODEL_BALANCED)
            chosen_local = catalog.get("selected_local_model")
            rationale.append("Deep reasoning detected → selecting reasoning-focused models")
        elif t == "analysis":
            chosen_cloud = best_cloud(cfg.CLOUD_MODEL_BALANCED, cfg.CLOUD_MODEL_REASON)
            chosen_local = catalog.get("selected_local_model")
            rationale.append("Analysis task → selecting balanced high-quality models")
        else:
            chosen_cloud = best_cloud(cfg.CLOUD_MODEL_FAST, cfg.CLOUD_MODEL_BALANCED)
            chosen_local = catalog.get("selected_local_model")
            rationale.append("General query → selecting fast, capable model")

        if chosen_local and chosen_local not in local_ids:
            chosen_local = list(local_ids)[0] if local_ids else None

        cloud_entry = self.resolve_cloud_model(chosen_cloud)
        return {
            "task_type":     t,
            "score":         task["score"],
            "local_model":   chosen_local,
            "cloud_model":   chosen_cloud,
            "cloud_entry":   cloud_entry,
            "mode":          catalog.get("execution_mode","local+cloud"),
            "rationale":     rationale,
        }

    # ─── provider status ──────────────────────────────────────

    def local_backend_status(self, refresh: bool = False) -> dict:
        """
        Returns status from the llama-cpp LocalModelManager.
        Falls back to Ollama if llama-cpp is not installed.
        """
        cache_key = "local_backend_status"
        if not refresh:
            cached = self._cached(cache_key)
            if cached is not None:
                return cached

        # ── Try llama-cpp first ────────────────────────────────
        mgr = _get_local_manager()
        if mgr is not None:
            try:
                status = mgr.status()
                status["backend"] = "llama_cpp"
                return self._store(cache_key, status)
            except Exception as e:
                logger.warning(f"LocalModelManager.status() failed: {e}")

        # ── Ollama fallback ────────────────────────────────────
        return self._store(cache_key, self._ollama_status_raw())

    def _ollama_status_raw(self) -> dict:
        """Ping Ollama directly (legacy fallback)."""
        url     = f"{_trim_slash(cfg.OLLAMA_BASE_URL)}/api/tags"
        started = time.perf_counter()
        try:
            resp = requests.get(url, timeout=5)
            latency_ms = _ms(started)
            resp.raise_for_status()
            models = []
            for m in resp.json().get("models",[]):
                details = m.get("details",{}) or {}
                size    = int(m.get("size") or 0)
                models.append({
                    "id":             m.get("name") or m.get("model"),
                    "name":           m.get("name") or m.get("model"),
                    "size":           size,
                    "size_gb":        round(size/(1024**3),2) if size else None,
                    "family":         details.get("family") or (details.get("families") or [None])[0],
                    "parameter_size": details.get("parameter_size"),
                    "modified_at":    m.get("modified_at"),
                    "provider":       "local",
                })
            return {
                "running":True,"connected":True,"latency_ms":latency_ms,
                "base_url":cfg.OLLAMA_BASE_URL,"models":models,"error":"",
                "backend":"ollama"
            }
        except Exception as e:
            return {
                "running":False,"connected":False,"latency_ms":_ms(started),
                "base_url":cfg.OLLAMA_BASE_URL,"models":[],"error":str(e),
                "backend":"ollama"
            }

    # Keep backward-compat alias
    def ollama_status(self, refresh: bool = False) -> dict:
        return self.local_backend_status(refresh=refresh)

    def _openrouter_headers(self, key: Optional[str] = None) -> dict:
        key = key if key is not None else self.openrouter_key()
        h = {"Content-Type":"application/json","HTTP-Referer":cfg.OPENROUTER_APP_URL,"X-Title":cfg.OPENROUTER_APP_NAME}
        if key: h["Authorization"] = f"Bearer {key}"
        return h

    def _is_free_openrouter_model(self, item: dict) -> bool:
        pricing = item.get("pricing") or {}
        return all(str(pricing.get(k,"1")) == "0" for k in ["prompt","completion","request"])

    def openrouter_status(self, refresh: bool = False) -> dict:
        if not refresh:
            cached = self._cached("openrouter_status")
            if cached is not None:
                return cached
        key = self.openrouter_key()
        # Skip network call entirely when no key is set — avoids 10s timeout on every cache miss
        if not key:
            return self._store("openrouter_status", {
                "configured": False, "connected": False, "latency_ms": 0,
                "base_url": self.openrouter_base(), "models": [], "free_models": [],
                "error": "No API key configured. Add one in Settings → Providers."
            })
        base    = self.openrouter_base()
        started = time.perf_counter()
        try:
            resp = requests.get(f"{base}/models", headers=self._openrouter_headers(key), timeout=10)
            latency_ms = _ms(started)
            resp.raise_for_status()
            models = []
            for item in resp.json().get("data",[]):
                model_id = item.get("id")
                if not model_id: continue
                models.append({
                    "id":                  model_id,
                    "name":                item.get("name") or model_id,
                    "provider":            "openrouter",
                    "context_length":      item.get("context_length"),
                    "free":                self._is_free_openrouter_model(item),
                    "pricing":             item.get("pricing") or {},
                    "supported_parameters": item.get("supported_parameters") or [],
                    "description":         item.get("description") or "",
                    "top_provider":        item.get("top_provider") or {},
                    "architecture":        item.get("architecture") or {},
                    "custom":              False,
                })
            free_models = [m for m in models if m.get("free")]
            return self._store("openrouter_status",{
                "configured":bool(key),"connected":True,"latency_ms":latency_ms,
                "base_url":base,"models":models,"free_models":free_models,"error":""
            })
        except Exception as e:
            return self._store("openrouter_status",{
                "configured":bool(key),"connected":False,"latency_ms":_ms(started),
                "base_url":base,"models":[],"free_models":[],"error":str(e)
            })

    def list_local_models(self, refresh: bool = False) -> list[dict]:
        return self.ollama_status(refresh=refresh).get("models",[])

    def list_cloud_models(self, refresh: bool = False) -> list[dict]:
        free_models = self.openrouter_status(refresh=refresh).get("free_models",[])
        custom      = self.custom_cloud_models()
        combined, seen = [], set()
        for item in free_models + custom:
            mid = item.get("id")
            if mid and mid not in seen:
                combined.append(item); seen.add(mid)
        return combined

    def resolve_cloud_model(self, model_id: Optional[str]) -> Optional[dict]:
        if not model_id: return None
        for item in self.custom_cloud_models():
            if item.get("id") == model_id or item.get("uid") == model_id:
                return item
        for item in self.list_cloud_models(refresh=False):
            if item.get("id") == model_id:
                return {**item,"base_url":self.openrouter_base(),"api_key":self.openrouter_key(),"provider":"openrouter","custom":False}
        return None

    def get_catalog(self, refresh: bool = False) -> dict:
        # Use the cache key so repeated calls during a single chat do not re-ping providers
        if not refresh:
            cached = self._cached("catalog")
            if cached is not None:
                return cached
        local_models   = self.list_local_models(refresh=refresh)
        cloud_models   = self.list_cloud_models(refresh=refresh)
        selected_local = self._get_pref("selected_local_model") or self._get_pref("local_model") or cfg.OLLAMA_MODEL_MAIN
        selected_cloud = self._get_pref("selected_cloud_model") or cfg.CLOUD_MODEL_BALANCED
        if local_models and selected_local not in {m.get("id") for m in local_models}:
            selected_local = local_models[0]["id"]
        if cloud_models and selected_cloud not in {m.get("id") for m in cloud_models}:
            selected_cloud = cloud_models[0]["id"]
        return self._store("catalog", {
            "execution_mode":       self.execution_mode(),
            "selected_local_model": selected_local,
            "selected_cloud_model": selected_cloud,
            "local_models":         local_models,
            "cloud_models":         cloud_models,
        })

    def status(self) -> dict:
        catalog    = self.get_catalog(refresh=False)
        ollama     = self.ollama_status(refresh=False)
        openrouter = self.openrouter_status(refresh=False)
        return {
            "local_running":        ollama.get("running",False),
            "cloud_available":      bool(self.openrouter_key()) and bool(catalog.get("cloud_models")),
            "cloud_provider":       "OpenRouter" if self.openrouter_key() else "None",
            "local_models":         [m.get("id") for m in catalog.get("local_models",[])],
            "execution_mode":       catalog.get("execution_mode"),
            "selected_local_model": catalog.get("selected_local_model"),
            "selected_cloud_model": catalog.get("selected_cloud_model"),
            "ollama_latency_ms":    ollama.get("latency_ms"),
            "openrouter_latency_ms": openrouter.get("latency_ms"),
        }

    def choose_route(self, message, attachments=None, execution_mode=None, local_model=None, cloud_model=None, intelligence_mode=False) -> dict:
        if intelligence_mode:
            return self.intelligence_select(message, attachments)
        task    = self.score_task(message, attachments or [])
        catalog = self.get_catalog(refresh=False)
        mode    = (execution_mode or catalog["execution_mode"] or cfg.DEFAULT_EXECUTION_MODE).lower()
        if mode not in {"local","cloud","local+cloud"}:
            mode = cfg.DEFAULT_EXECUTION_MODE
        local_models  = catalog.get("local_models",[])
        local_ids     = {m.get("id") for m in local_models}
        selected_local = local_model or catalog.get("selected_local_model")
        if task["type"] == "coding" and cfg.OLLAMA_MODEL_CODE in local_ids and not local_model:
            selected_local = cfg.OLLAMA_MODEL_CODE
        if selected_local not in local_ids and local_models:
            selected_local = local_models[0]["id"]
        if not local_models:
            selected_local = None
        cloud_entry = self.resolve_cloud_model(cloud_model or catalog.get("selected_cloud_model"))
        if task["type"] == "coding" and not cloud_model:
            cloud_entry = self.resolve_cloud_model(cfg.CLOUD_MODEL_CODING) or cloud_entry
        if task["type"] in {"analysis","creative"} and not cloud_model:
            cloud_entry = self.resolve_cloud_model(cfg.CLOUD_MODEL_BALANCED) or cloud_entry
        if task["type"] == "reasoning" and not cloud_model:
            cloud_entry = self.resolve_cloud_model(cfg.CLOUD_MODEL_REASON) or cloud_entry
        selected_cloud = cloud_entry.get("id") if cloud_entry else None

        # Check if cloud is actually usable (key present)
        has_cloud_key = bool(
            (cloud_entry and cloud_entry.get("api_key")) or self.openrouter_key()
        )
        effective_cloud = selected_cloud if has_cloud_key else None

        if mode == "local"       and not selected_local and effective_cloud: mode = "cloud"
        if mode == "cloud"       and not effective_cloud and selected_local:  mode = "local"
        if mode == "local+cloud":
            if not selected_local and effective_cloud:     mode = "cloud"
            elif not effective_cloud and selected_local:   mode = "local"
        return {
            "mode":        mode,
            "task_type":   task["type"],
            "score":       task["score"],
            "local_model": selected_local,
            "cloud_model": selected_cloud,
            "cloud_entry": cloud_entry,
        }

    # ─── chat methods ─────────────────────────────────────────

    def chat_local(self, messages, model, *, fmt=None, timeout=None) -> ChatResult:
        """
        Chat with a local model.
        Uses llama-cpp-python via LocalModelManager when available,
        falls back to Ollama HTTP API.
        """
        if not model:
            raise RuntimeError("No local model selected.")

        # ── Try llama-cpp path ─────────────────────────────────
        mgr = _get_local_manager()
        if mgr is not None:
            try:
                started = time.perf_counter()
                result  = mgr.chat(messages, model, max_tokens=1800, temperature=0.7)
                return ChatResult(
                    "local", model,
                    result.get("content", ""),
                    result.get("latency_ms", _ms(started)),
                    result
                )
            except Exception as e:
                logger.warning(f"llama-cpp chat failed ({e}), falling back to Ollama")

        # ── Ollama fallback ────────────────────────────────────
        payload: dict[str,Any] = {"model":model,"messages":messages,"stream":False}
        if fmt: payload["format"] = fmt
        started = time.perf_counter()
        local_to = timeout or getattr(cfg,'LOCAL_TIMEOUT_SECS', cfg.REQUEST_TIMEOUT_SECS)
        resp = requests.post(
            f"{_trim_slash(cfg.OLLAMA_BASE_URL)}/api/chat",
            json=payload, timeout=local_to
        )
        latency_ms = _ms(started)
        try: data = resp.json()
        except Exception: data = {"raw_text": resp.text}
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama error {resp.status_code}: {data.get('error') if isinstance(data,dict) else resp.text}")
        if isinstance(data,dict) and data.get("total_duration"):
            latency_ms = round(data["total_duration"]/1_000_000,2)
        return ChatResult("local", model, ((data or {}).get("message") or {}).get("content",""), latency_ms, data)

    def _normalize_openai_base(self, base_url: str) -> str:
        base = _trim_slash(base_url)
        return base[:-len('/chat/completions')] if base.endswith('/chat/completions') else base

    def chat_cloud(self, messages, model, *, base_url=None, api_key=None, timeout=None, max_tokens=1800) -> ChatResult:
        base = self._normalize_openai_base(base_url or self.openrouter_base())
        key  = (api_key or self.openrouter_key()).strip()
        if not key:
            raise RuntimeError("No cloud API key configured.")
        started = time.perf_counter()
        cloud_to = timeout or getattr(cfg,'CLOUD_TIMEOUT_SECS', 45)
        resp = requests.post(
            f"{base}/chat/completions",
            headers=self._openrouter_headers(key),
            json={"model":model,"messages":messages,"temperature":0.3,"max_tokens":max_tokens},
            timeout=cloud_to
        )
        latency_ms = _ms(started)
        try: data = resp.json()
        except Exception: data = {"raw_text": resp.text}
        if resp.status_code >= 400:
            err_msg = data.get("error") if isinstance(data,dict) else resp.text
            raise RuntimeError(f"Cloud error {resp.status_code}: {err_msg}")
        choices = data.get("choices") or []
        content = (choices[0].get("message") or {}).get("content","") if choices else ""
        return ChatResult("cloud", model, content, latency_ms, data)

    def test_model(self, provider, model, base_url="", api_key="") -> dict:
        started = time.perf_counter()
        try:
            if provider == "local":
                result = self.chat_local([{"role":"user","content":"Reply with the single word: pong"}], model, timeout=cfg.MODEL_TEST_TIMEOUT_SECS)
            else:
                result = self.chat_cloud([{"role":"user","content":"Reply with the single word: pong"}], model,
                    base_url=base_url or self.openrouter_base(), api_key=api_key or self.openrouter_key(),
                    timeout=cfg.MODEL_TEST_TIMEOUT_SECS, max_tokens=8)
            return {"ok":True,"provider":result.provider,"model":result.model,"latency_ms":result.latency_ms or _ms(started),"response_preview":(result.content or "")[:120],"error":""}
        except Exception as e:
            return {"ok":False,"provider":provider,"model":model,"latency_ms":_ms(started),"response_preview":"","error":str(e)}
