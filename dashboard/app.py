# ============================================================
#  Dashboard API v6 — Aria Personal Agent
# ============================================================

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import config as cfg

app = FastAPI(title="Aria Personal Agent v6")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_pool   = ThreadPoolExecutor(max_workers=10, thread_name_prefix="aria-worker")
_agent  = None
_monitor = None
_sync   = None


def set_agent(agent, monitor=None, sync=None):
    global _agent, _monitor, _sync
    _agent, _monitor, _sync = agent, monitor, sync


async def _run(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, lambda: fn(*args, **kwargs))


def _need_agent():
    if not _agent:
        raise HTTPException(503, "Agent is still starting. Please retry in a moment.")
    return _agent


def _secret_mask(key: str, value: Any) -> Any:
    if value in (None,""): return value
    key = (key or "").lower()
    if any(t in key for t in ["key","token","secret","password"]):
        text = str(value)
        return f"***{text[-4:]}" if len(text) >= 4 else "***"
    return value


_ENV_KEY_MAP = {
    "openrouter_key": "OPENROUTER_API_KEY",
    "supabase_url":   "SUPABASE_URL",
    "supabase_key":   "SUPABASE_KEY",
}


def _save_pref(key: str, value: Any):
    agent = _need_agent()
    value = "" if value is None else str(value)
    agent.set_preference(key, value)
    env_key = _ENV_KEY_MAP.get(key)
    if env_key is not None:
        os.environ[env_key] = value
    if key.startswith("supabase_") and _sync:
        try: _sync.reconfigure()
        except Exception: pass
    return True


def _artifact_payload(item: dict) -> dict:
    filepath = item.get("filepath") or item.get("path") or ""
    path     = Path(filepath) if filepath else None
    modified = ""
    if path and path.exists():
        import datetime as _dt
        modified = _dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    name    = item.get("filename") or item.get("name") or (path.name if path else "")
    created = item.get("created_at") or item.get("modified") or modified or ""
    return {
        "id":           item.get("id") or name,
        "name":         name,
        "path":         filepath,
        "filetype":     item.get("filetype") or (path.suffix.lower() if path else ""),
        "size":         int(item.get("size") or (path.stat().st_size if path and path.exists() else 0)),
        "created_at":   created,
        "modified":     modified or created,
        "model_used":   item.get("model_used") or "",
        "source_prompt": item.get("source_prompt") or "",
        "download_url": f"/api/download/{name}",
    }


def _local_and_cloud_cards(refresh: bool = False) -> dict:
    agent    = _need_agent()
    router   = agent.router
    catalog  = router.get_catalog(refresh=refresh)
    ollama   = router.ollama_status(refresh=refresh)
    openrouter = router.openrouter_status(refresh=refresh)
    local_models = []
    for item in catalog.get("local_models",[]):
        cap = cfg.MODEL_CAPABILITIES.get(item.get("id") or "")
        local_models.append({
            **item,
            "selected":  item.get("id") == catalog.get("selected_local_model"),
            "connected": bool(ollama.get("connected")),
            "latency_ms": ollama.get("latency_ms"),
            "status":    "ready" if ollama.get("connected") else "offline",
            "capabilities": cap or {},
        })
    cloud_models = []
    for item in catalog.get("cloud_models",[]):
        provider  = item.get("provider") or ("custom" if item.get("custom") else "openrouter")
        connected = bool(openrouter.get("connected")) if provider == "openrouter" else bool(item.get("base_url"))
        cap       = cfg.MODEL_CAPABILITIES.get(item.get("id") or "")
        cloud_models.append({
            **item,
            "selected":   item.get("id") == catalog.get("selected_cloud_model"),
            "connected":  connected,
            "latency_ms": openrouter.get("latency_ms") if provider == "openrouter" else None,
            "status":     "ready" if connected else "needs setup",
            "capabilities": cap or {},
        })
    return {**catalog, "local_models":local_models, "cloud_models":cloud_models, "ollama":ollama, "openrouter":openrouter}


# ─── Pydantic models ─────────────────────────────────────────

class ChatRequest(BaseModel):
    message:              str
    attached_file_ids:    list[str]      = Field(default_factory=list)
    execution_mode:       Optional[str]  = None
    local_model:          Optional[str]  = None
    cloud_model:          Optional[str]  = None
    intelligence_mode:    bool           = False

class TaskCreate(BaseModel):
    title:            str
    description:      str           = ""
    priority:         str           = "normal"
    scheduled_at:     Optional[str] = None
    reminder_minutes: int           = 15

class TaskUpdate(BaseModel):
    title:            Optional[str] = None
    description:      Optional[str] = None
    status:           Optional[str] = None
    priority:         Optional[str] = None
    scheduled_at:     Optional[str] = None
    reminder_minutes: Optional[int] = None
    result:           Optional[str] = None

class TaskRunRequest(BaseModel):
    execution_mode: Optional[str] = None
    local_model:    Optional[str] = None
    cloud_model:    Optional[str] = None

class FactRequest(BaseModel):
    key:   str
    value: str

class PrefRequest(BaseModel):
    key:   Optional[str]       = None
    value: Optional[str]       = None
    items: dict[str,Any]       = Field(default_factory=dict)

class ModelTestRequest(BaseModel):
    provider:  str
    model:     str
    base_url:  str = ""
    api_key:   str = ""

class CustomCloudModelRequest(BaseModel):
    name:        str  = ""
    model:       str
    base_url:    str  = ""
    api_key:     str  = ""
    description: str  = ""
    free:        bool = True

class AdminVerifyRequest(BaseModel):
    password: str

class AdminSettingsRequest(BaseModel):
    agent_name:          Optional[str] = None
    agent_voice_gender:  Optional[str] = None
    agent_voice_name:    Optional[str] = None
    admin_password:      Optional[str] = None  # new password to set
    new_password_hash:   Optional[str] = None


# ─── Routes ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    file = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(file.read_text(encoding="utf-8"))

@app.get("/api/ping")
async def ping():
    return {"ok": True, "agent_ready": _agent is not None, "agent_name": cfg.AGENT_NAME, "version": cfg.AGENT_VERSION}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    agent = _need_agent()
    if not (req.message or "").strip():
        raise HTTPException(400, "Message is required.")
    files = []
    if req.attached_file_ids:
        uploads = agent.memory.structured.get_uploads(limit=500)
        for fid in req.attached_file_ids:
            match = next((u for u in uploads if u["id"] == fid), None)
            if match: files.append(match["filepath"])
    # Support intelligence_mode flag
    if req.intelligence_mode:
        result = await _run(
            agent.chat, req.message.strip(), files, False,
            req.execution_mode, req.local_model, req.cloud_model
        )
        # Inject intelligence route info
        intel = agent.router.intelligence_select(req.message, files)
        result["intelligence_route"] = intel
    else:
        result = await _run(
            agent.chat, req.message.strip(), files, False,
            req.execution_mode, req.local_model, req.cloud_model
        )
    return JSONResponse(result)

@app.post("/api/intelligence/route")
async def intelligence_route(req: ChatRequest):
    """Preview which models intelligence mode would select for a given prompt."""
    agent = _need_agent()
    files = []
    if req.attached_file_ids:
        uploads = agent.memory.structured.get_uploads(limit=500)
        for fid in req.attached_file_ids:
            match = next((u for u in uploads if u["id"] == fid), None)
            if match: files.append(match["filepath"])
    route = agent.router.intelligence_select(req.message, files)
    return JSONResponse(route)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), add_to_knowledge: str = Form("false")):
    agent       = _need_agent()
    to_knowledge = add_to_knowledge.lower() in {"true","1","yes"}
    uid         = str(uuid.uuid4())[:8]
    filename    = file.filename or f"upload-{uid}"
    dest_dir    = Path(cfg.DOCUMENTS_DIR if to_knowledge else cfg.UPLOADS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest        = dest_dir / f"{uid}_{Path(filename).name}"
    content     = await file.read()
    dest.write_bytes(content)
    agent.memory.structured.add_upload(uid, filename, str(dest), len(content), dest.suffix.lower())
    indexed = False
    if to_knowledge:
        def _ingest():
            from agent.file_processor import read_file_for_llm
            from ingestion.ingest import chunk_text
            text = read_file_for_llm(str(dest))
            if not text or text.startswith("Error"): return False
            for idx, chunk in enumerate(chunk_text(text)):
                agent.memory.vector.save_document(chunk, filename, idx)
            agent.memory.structured.mark_indexed(uid)
            return True
        indexed = await _run(_ingest)
    return {"id":uid,"filename":filename,"size":len(content),"indexed":indexed,"knowledge":to_knowledge}

@app.get("/api/uploads")
async def list_uploads(limit: int = 100):
    return _need_agent().memory.structured.get_uploads(limit=limit)

@app.get("/api/generated")
async def list_generated(limit: int = 200):
    agent         = _need_agent()
    generated_dir = Path(cfg.GENERATED_DIR)
    rows          = agent.memory.structured.get_artifacts(limit=limit)
    seen, items   = set(), []
    for row in rows:
        payload = _artifact_payload(row)
        if payload["name"]: seen.add(payload["name"]); items.append(payload)
    if generated_dir.exists():
        for fp in sorted(generated_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if fp.is_file() and fp.name not in seen:
                items.append(_artifact_payload({
                    "id":fp.name,"filename":fp.name,"filepath":str(fp),
                    "filetype":fp.suffix.lower(),"size":fp.stat().st_size,
                    "created_at":"","model_used":"","source_prompt":"",
                }))
                seen.add(fp.name)
    items.sort(key=lambda x: x.get("modified") or x.get("created_at") or "", reverse=True)
    return items[:limit]

@app.get("/api/download/{filename:path}")
async def download_file(filename: str):
    path = Path(cfg.GENERATED_DIR) / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), filename=path.name)

# ─── Ollama management ────────────────────────────────────────

@app.get("/api/ollama/status")
@app.get("/api/ollama/status")
async def ollama_status():
    _need_agent()
    # Run in thread pool — makes a blocking HTTP request
    return await _run(_need_agent().router.ollama_status, True)

@app.post("/api/ollama/start")
async def ollama_start():
    """Try to start Ollama server if not running."""
    import platform, time as _time

    def _do_start():
        status = _need_agent().router.ollama_status(refresh=True)
        if status.get("connected"):
            return {"ok": True, "message": "Ollama is already running.", "status": status}
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(["ollama","serve"], creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess,"CREATE_NO_WINDOW") else 0)
            else:
                subprocess.Popen(["ollama","serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _time.sleep(3)
            status = _need_agent().router.ollama_status(refresh=True)
            msg = "Ollama started successfully." if status.get("connected") else "Ollama start attempted, please wait a few seconds and try again."
            return {"ok": status.get("connected", False), "message": msg, "status": status}
        except FileNotFoundError:
            return {"ok": False, "message": "Ollama is not installed. Download from https://ollama.ai", "status": status}
        except Exception as e:
            return {"ok": False, "message": f"Failed to start Ollama: {e}", "status": status}

    return await _run(_do_start)

# ─── Tasks ────────────────────────────────────────────────────

@app.get("/api/tasks")
async def get_tasks(status: Optional[str] = Query(None)):
    return _need_agent().get_tasks(status)

@app.post("/api/tasks")
async def create_task(req: TaskCreate):
    agent   = _need_agent()
    task_id = agent.add_task(req.title, req.description, req.priority, req.scheduled_at, req.reminder_minutes)
    task    = next((t for t in agent.get_tasks() if t["id"] == task_id), None)
    return task or {"id": task_id}

@app.put("/api/tasks/{task_id}")
async def update_task(task_id: str, req: TaskUpdate):
    agent   = _need_agent()
    payload = req.model_dump(exclude_none=True)
    if not payload: raise HTTPException(400,"Nothing to update.")
    agent.update_task(task_id, **payload)
    return {"ok":True,"task":next((t for t in agent.get_tasks() if t["id"]==task_id),None)}

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    _need_agent().delete_task(task_id)
    return {"deleted": task_id}

@app.post("/api/tasks/{task_id}/run")
async def run_task(task_id: str, req: TaskRunRequest):
    result = await _run(_need_agent().run_task, task_id, req.execution_mode, req.local_model, req.cloud_model)
    return JSONResponse(result)

@app.get("/api/tasks/reminders")
async def task_reminders():
    return _need_agent().get_task_reminders()

# ─── Facts & preferences ─────────────────────────────────────

@app.get("/api/facts")
async def get_facts():
    return _need_agent().memory.structured.get_all_facts()

@app.post("/api/facts")
async def set_fact(req: FactRequest):
    _need_agent().learn_fact(req.key, req.value)
    return {"saved": True}

@app.get("/api/knowledge/sources")
async def knowledge_sources():
    return _need_agent().memory.vector.get_sources()

@app.get("/api/preferences")
async def get_preferences():
    agent = _need_agent()
    prefs = agent.memory.structured.get_preferences()
    or_key = prefs.get("openrouter_key","") or os.getenv("OPENROUTER_API_KEY","")
    sb_url = prefs.get("supabase_url","") or os.getenv("SUPABASE_URL","")
    sb_key = prefs.get("supabase_key","") or os.getenv("SUPABASE_KEY","")
    return {
        # Masked for display — use has_* flags to know whether a key is actually set
        "preferences": {k: _secret_mask(k,v) for k,v in prefs.items()},
        # Boolean indicators for UI
        "has_openrouter_key": bool(or_key.strip()),
        "has_supabase":       bool(sb_url.strip() and sb_key.strip()),
        # Expose last-4 of key so user can confirm which key is saved
        "openrouter_key_hint": f"…{or_key.strip()[-4:]}" if len(or_key.strip()) >= 4 else ("(set)" if or_key.strip() else ""),
        "agent_name":         cfg.AGENT_NAME,
        "agent_voice_gender": cfg.AGENT_VOICE_GENDER,
        "agent_voice_name":   cfg.AGENT_VOICE_NAME,
    }

@app.post("/api/preferences")
async def set_preferences(req: PrefRequest):
    agent  = _need_agent()
    items  = dict(req.items or {})
    if req.key: items[req.key] = req.value or ""
    if not items: raise HTTPException(400, "No preferences provided.")

    _SECRET_KEYS = {"openrouter_key","supabase_key","groq_api_key","admin_password_hash"}

    def _do_save():
        saved = []
        for key, value in items.items():
            str_val = str(value).strip() if value is not None else ""
            # Never overwrite a saved secret with blank — user just left the field empty
            if key in _SECRET_KEYS and not str_val:
                existing = agent.memory.structured.get_preference(key, "")
                if existing:
                    continue  # skip — keep existing secret
            _save_pref(key, str_val)
            saved.append(key)
        agent.router.refresh()
        return saved

    saved = await _run(_do_save)
    return {"saved": saved}

# ─── Models ──────────────────────────────────────────────────

@app.get("/api/models")
async def get_models(refresh: bool = False):
    _need_agent()
    # Run in thread pool — ollama_status + openrouter_status make blocking HTTP calls
    data = await _run(_local_and_cloud_cards, refresh)
    return data

@app.post("/api/models/refresh")
async def refresh_models():
    """Refresh model list in thread pool so it does not block the event loop."""
    agent = _need_agent()
    # Clear router cache so next status check re-fetches
    agent.router.refresh()
    # Run the potentially slow OpenRouter fetch in the thread executor
    result = await _run(_local_and_cloud_cards, True)
    return result

@app.post("/api/models/test")
async def test_model(req: ModelTestRequest):
    return await _run(_need_agent().router.test_model, req.provider, req.model, req.base_url, req.api_key)

@app.post("/api/models/custom")
async def add_custom_model(req: CustomCloudModelRequest):
    entry = _need_agent().router.add_custom_cloud_model(req.name, req.model, req.base_url or cfg.OPENROUTER_BASE, req.api_key, req.description, req.free)
    return {"saved":True,"entry":entry}

@app.delete("/api/models/custom/{uid}")
async def delete_custom_model(uid: str):
    removed = _need_agent().router.remove_custom_cloud_model(uid)
    return {"removed":removed,"uid":uid}

@app.get("/api/models/capabilities")
async def model_capabilities():
    """Return capability catalogue for all known models."""
    agent   = _need_agent()
    catalog = agent.router.get_catalog(refresh=False)
    result  = {}
    # Merge known caps with live models
    for m in catalog.get("local_models",[]) + catalog.get("cloud_models",[]):
        mid = m.get("id")
        if not mid: continue
        cap = dict(cfg.MODEL_CAPABILITIES.get(mid) or {})
        cap["provider"]    = m.get("provider","")
        cap["name"]        = m.get("name","")
        cap["description"] = m.get("description","")
        cap["size_gb"]     = m.get("size_gb")
        cap["context_length"] = m.get("context_length")
        if not cap.get("strengths"): cap["strengths"] = []
        if not cap.get("best_for"):  cap["best_for"]  = m.get("description","")[:100]
        result[mid] = cap
    # Also include known caps for models not yet pulled
    for mid, cap in cfg.MODEL_CAPABILITIES.items():
        if mid not in result:
            result[mid] = {**cap, "provider":"not_installed"}
    return result

# ─── Connections ─────────────────────────────────────────────

@app.get("/api/connections")
async def connections(refresh: bool = False):
    agent  = _need_agent()
    router = agent.router

    def _get_connections():
        sync_status = _sync.sync_status() if _sync else {
            "enabled":False,"configured":bool(cfg.SUPABASE_URL and cfg.SUPABASE_KEY),
            "connected":False,"provider":"Supabase","device":cfg.DEVICE_NAME,
            "last_sync_at":None,"last_error":"Sync manager not running.",
            "push_count":0,"pull_count":0,
        }
        sync_logs = agent.memory.structured.get_sync_logs(limit=20)
        last_sync_latency = next((r.get("latency_ms") for r in sync_logs if r.get("latency_ms") is not None),None)
        sync_status["latency_ms"] = last_sync_latency
        return {
            "ollama":    router.ollama_status(refresh=refresh),
            "openrouter": router.openrouter_status(refresh=refresh),
            "supabase":  sync_status,
            "catalog":   router.get_catalog(refresh=refresh),
            "sync_logs": sync_logs,
        }

    # Run blocking network checks in the thread pool
    return await _run(_get_connections)

@app.get("/api/sync/logs")
async def sync_logs(limit: int = 80):
    return _need_agent().memory.structured.get_sync_logs(limit)

@app.post("/api/sync/run")
async def run_sync():
    if not _sync: raise HTTPException(503,"Sync manager is not running.")
    return await _run(_sync.sync_now)

@app.get("/api/stats")
async def stats():
    agent   = _need_agent()
    payload = await _run(agent.stats)
    payload["agent_name"]    = cfg.AGENT_NAME
    payload["agent_version"] = cfg.AGENT_VERSION
    payload["device"]        = cfg.DEVICE_NAME
    # Count generated files from DB artifacts (fast) — avoid slow filesystem scan
    try:
        payload["generated_files"] = len(agent.memory.structured.get_artifacts(500))
    except Exception:
        payload["generated_files"] = 0
    if _monitor: payload["monitor"] = _monitor.get_status()
    if _sync:    payload["sync"]    = _sync.sync_status()
    return payload

@app.get("/api/logs")
async def logs(limit: int = 80):
    return _need_agent().memory.structured.get_logs(limit)

@app.get("/api/suggestions")
async def suggestions():
    if not _monitor: return []
    return _monitor.get_suggestions()

@app.get("/api/activity")
async def activity(days: int = 7):
    return _need_agent().memory.structured.get_activity_summary(days)

# ─── Admin / Advanced ────────────────────────────────────────

@app.post("/api/admin/verify")
async def admin_verify(req: AdminVerifyRequest):
    """Verify admin password - does not require agent to be fully ready."""
    try:
        pw_hash  = hashlib.sha256(req.password.encode()).hexdigest()
        cfg_hash = getattr(cfg, "ADMIN_PASSWORD_HASH", None)
        ok       = bool(cfg_hash and pw_hash == cfg_hash)
        if _agent:
            try:
                stored = _agent.memory.structured.get_preference("admin_password_hash")
                if stored:
                    ok = (pw_hash == stored)
            except Exception:
                pass
        return {"ok": ok}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/admin/settings")
async def admin_get_settings():
    """Get admin-controlled settings (requires prior verification on client side)."""
    agent = _need_agent()
    return {
        "agent_name":         cfg.AGENT_NAME,
        "agent_voice_gender": cfg.AGENT_VOICE_GENDER,
        "agent_voice_name":   cfg.AGENT_VOICE_NAME,
        "hybrid_cloud_first": cfg.HYBRID_CLOUD_FIRST,
    }

@app.post("/api/admin/settings")
async def admin_save_settings(req: AdminSettingsRequest):
    """Save admin settings."""
    agent  = _need_agent()
    saved  = []
    if req.agent_name:
        cfg.AGENT_NAME = req.agent_name
        agent.set_preference("agent_name", req.agent_name)
        os.environ["AGENT_NAME"] = req.agent_name
        saved.append("agent_name")
    if req.agent_voice_gender in ("male","female",""):
        cfg.AGENT_VOICE_GENDER = req.agent_voice_gender or "female"
        agent.set_preference("agent_voice_gender", cfg.AGENT_VOICE_GENDER)
        saved.append("agent_voice_gender")
    if req.agent_voice_name is not None:
        cfg.AGENT_VOICE_NAME = req.agent_voice_name
        agent.set_preference("agent_voice_name", req.agent_voice_name)
        saved.append("agent_voice_name")
    if req.new_password_hash:
        agent.set_preference("admin_password_hash", req.new_password_hash)
        saved.append("admin_password")
    return {"saved": saved}

# ─── HuggingFace / Local Model Manager API ───────────────────

# Module-level singleton — all API calls share one engine instance
_lmm_instance = None
_lmm_init_lock = None
_lmm_init_error = None

def _init_lmm_background():
    """Init LocalModelManager in background — called once at startup."""
    global _lmm_instance, _lmm_init_error
    try:
        from pathlib import Path as _Path
        _Path(cfg.DATA_DIR).mkdir(parents=True, exist_ok=True)
        from agent.local_model_manager import LocalModelManager
        _lmm_instance = LocalModelManager()
    except Exception as e:
        _lmm_init_error = str(e)
        import logging
        logging.getLogger("aria.dashboard").warning(f"LocalModelManager init: {e}")


# Kick off background init as soon as the module loads — ready by first request
import threading as _bg_thread
_bg_thread.Thread(target=_init_lmm_background, daemon=True, name="lmm-init").start()


def _get_lmm():
    """Return the shared LocalModelManager singleton.
    Returns quickly — raises 503 only if init actually failed (not still pending)."""
    global _lmm_instance, _lmm_init_error
    if _lmm_instance is not None:
        return _lmm_instance
    # Still initialising — wait up to 5 s then report status
    import time
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if _lmm_instance is not None:
            return _lmm_instance
        if _lmm_init_error is not None:
            break
        time.sleep(0.1)
    if _lmm_instance is not None:
        return _lmm_instance
    raise HTTPException(503, detail={
        "error": "Local model manager unavailable",
        "reason": _lmm_init_error or "Still initialising — try again in a moment",
        "hint": "Run: pip install llama-cpp-python huggingface-hub"
    })


class HFDownloadRequest(BaseModel):
    repo_id:     str
    filename:    str
    hf_token:    str = ""
    name:        str = ""
    description: str = ""


class HFImportRequest(BaseModel):
    file_path:   str
    name:        str = ""
    description: str = ""


class ModelSettingsRequest(BaseModel):
    model_id:    str
    gpu_layers:  Optional[int] = None
    context_len: Optional[int] = None
    name:        Optional[str] = None


class ModelLoadRequest(BaseModel):
    model_id: str


class ModelRemoveRequest(BaseModel):
    model_id:    str
    delete_file: bool = False


@app.get("/api/hf/system")
async def hf_system_info(refresh: bool = False):
    """Return hardware capabilities and inference recommendations."""
    def _get():
        try:
            mgr  = _get_lmm()
            caps = mgr.get_capabilities(refresh=refresh)
            return caps.to_dict()
        except HTTPException:
            # llama-cpp not installed — still run system analysis standalone
            try:
                from agent.local_model_manager import SystemAnalyser
                caps = SystemAnalyser.analyse()
                d = caps.to_dict()
                d["backend_available"] = False
                return d
            except Exception as e2:
                return {
                    "platform": "", "cpu_cores": 0, "cpu_threads": 0,
                    "ram_total_gb": 0, "ram_available_gb": 0,
                    "gpu_available": False, "vram_total_gb": 0, "vram_free_gb": 0,
                    "recommended_quant": "Q4_K_M", "max_model_size_gb": 4.0,
                    "backend_available": False, "error": str(e2)
                }
    return await _run(_get)


@app.get("/api/hf/status")
async def hf_status():
    """Quick connectivity check — always returns 200, never raises."""
    try:
        from agent.local_model_manager import LLAMA_CPP_AVAILABLE, HF_AVAILABLE
        lmm_ok = _lmm_instance is not None
        return {
            "ok": True,
            "llama_cpp_installed": LLAMA_CPP_AVAILABLE,
            "huggingface_hub_installed": HF_AVAILABLE,
            "manager_ready": lmm_ok,
            "error": _lmm_init_error,
        }
    except Exception as e:
        return {"ok": True, "llama_cpp_installed": False,
                "huggingface_hub_installed": False,
                "manager_ready": False, "error": str(e)}


@app.get("/api/hf/models")
async def hf_list_models():
    """List all locally registered GGUF models."""
    def _get():
        try:
            mgr = _get_lmm()
            return {
                "models":        mgr.list_local_models(),
                "active_model":  mgr.engine.current_model,
                "backend_ready": mgr.engine.is_loaded,
                "backend":       "llama_cpp",
                "error":         None,
            }
        except HTTPException as e:
            return {
                "models":        [],
                "active_model":  "",
                "backend_ready": False,
                "backend":       "unavailable",
                "error":         e.detail if isinstance(e.detail, str) else str(e.detail),
            }
    return await _run(_get)


@app.get("/api/hf/search")
async def hf_search_models(q: str = "", limit: int = 30, tags: str = ""):
    """Search HuggingFace Hub for GGUF models."""
    def _get():
        mgr  = _get_lmm()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        return {"results": mgr.search_models(query=q, limit=limit, tags=tag_list)}
    return await _run(_get)


@app.get("/api/hf/repo/files")
async def hf_repo_files(repo_id: str):
    """List GGUF files in a HuggingFace repo with sizes and quant info."""
    def _get():
        mgr = _get_lmm()
        return {"files": mgr.get_repo_files(repo_id), "repo_id": repo_id}
    return await _run(_get)


@app.post("/api/hf/download")
async def hf_download_model(req: HFDownloadRequest):
    """Start downloading a model from HuggingFace."""
    def _get():
        hf_token = req.hf_token or getattr(cfg, "HF_TOKEN", "")
        mgr = _get_lmm()
        return mgr.download_model(
            req.repo_id, req.filename,
            hf_token=hf_token, name=req.name, description=req.description
        )
    return await _run(_get)


@app.get("/api/hf/downloads")
async def hf_download_status():
    """List all active and recent downloads."""
    def _get():
        return {"downloads": _get_lmm().list_downloads()}
    return await _run(_get)


@app.get("/api/hf/downloads/{task_id}")
async def hf_download_progress(task_id: str):
    """Get progress for a specific download task."""
    return _get_lmm().download_progress(task_id)


@app.post("/api/hf/import")
async def hf_import_local(req: HFImportRequest):
    """Register an existing .gguf file from the local filesystem."""
    def _do():
        return _get_lmm().import_local_file(req.file_path, req.name, req.description)
    return await _run(_do)


@app.post("/api/hf/load")
async def hf_load_model(req: ModelLoadRequest):
    """Load a model into the llama-cpp inference engine."""
    def _do():
        return _get_lmm().load_model(req.model_id)
    return await _run(_do)


@app.post("/api/hf/unload")
async def hf_unload_model():
    """Unload the currently loaded model, freeing memory."""
    def _do():
        _get_lmm().unload_model()
        return {"ok": True, "message": "Model unloaded"}
    return await _run(_do)


@app.post("/api/hf/settings")
async def hf_update_settings(req: ModelSettingsRequest):
    """Update GPU layers / context window / name for a model."""
    def _do():
        return _get_lmm().update_model_settings(
            req.model_id,
            gpu_layers=req.gpu_layers,
            context_len=req.context_len,
            name=req.name,
        )
    return await _run(_do)


@app.post("/api/hf/remove")
async def hf_remove_model(req: ModelRemoveRequest):
    """Remove a model from the registry (optionally delete file)."""
    def _do():
        return _get_lmm().remove_model(req.model_id, delete_file=req.delete_file)
    return await _run(_do)


@app.get("/api/hf/recommend")
async def hf_recommend(model_size_gb: float = 7.0):
    """Get GPU layer / context recommendations for a model of given size."""
    def _do():
        from agent.local_model_manager import SystemAnalyser
        mgr  = _get_lmm()
        caps = mgr.get_capabilities()
        rec  = SystemAnalyser.recommend_for_model(model_size_gb, caps)
        return {"recommendation": rec, "capabilities": caps.to_dict()}
    return await _run(_do)


@app.get("/hf")
async def hf_manager_page():
    """Serve the HuggingFace model manager page.
    Uses FileResponse so FastAPI handles encoding — no manual read_text()."""
    from fastapi.responses import FileResponse, HTMLResponse
    tmpl = Path(__file__).parent / "templates" / "hf_manager.html"
    if tmpl.exists():
        # FileResponse reads as binary and sets correct Content-Type header
        return FileResponse(str(tmpl), media_type="text/html; charset=utf-8")
    return HTMLResponse(
        "<html><body style='font-family:sans-serif;padding:40px'>"
        "<h2>&#x1F6A7; HF Model Manager</h2>"
        "<p>Template file not found at:<br><code>" + str(tmpl) + "</code></p>"
        "<p>Make sure <b>hf_manager.html</b> is in <b>dashboard/templates/</b></p>"
        "</body></html>"
    )
