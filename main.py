#!/usr/bin/env python3
# ============================================================
#  Aria Personal Agent v6 — Main Entry Point
#  Run:  python main.py
#  Then open: http://localhost:8000
# ============================================================

import sys
import threading
import argparse
import subprocess
import time
import platform
from pathlib import Path

parser = argparse.ArgumentParser(description="Aria Personal Agent v6")
parser.add_argument("--no-dashboard",  action="store_true")
parser.add_argument("--no-scheduler",  action="store_true")
parser.add_argument("--no-monitor",    action="store_true")
parser.add_argument("--no-sync",       action="store_true")
parser.add_argument("--ingest",        action="store_true", help="Ingest docs and exit")
parser.add_argument("--model",         type=str)
parser.add_argument("--port",          type=int, default=8000)
args = parser.parse_args()

if args.model:
    import config
    config.OLLAMA_MODEL_MAIN = config.OLLAMA_MODEL_FAST = args.model

import config as cfg

if args.port != 8000:
    cfg.DASHBOARD_PORT = args.port


def banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  ✦  {cfg.AGENT_NAME} Personal Agent v7                     ║")
    print("║     HuggingFace + llama-cpp · Cloud · Private        ║")
    print("║     LM Studio-style local model management ⚡        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def ensure_local_backend():
    """
    Check local inference backend availability.
    Priority: llama-cpp-python > Ollama fallback.
    """
    import requests

    # 1. llama-cpp-python (preferred)
    try:
        import llama_cpp  # noqa
        print("  [Backend] llama-cpp-python ✓ — direct HuggingFace GGUF inference")
        try:
            from agent.local_model_manager import LocalModelManager
            mgr    = LocalModelManager()
            models = mgr.list_local_models()
            if models:
                print(f"  [Models]  {len(models)} local model(s) registered")
            else:
                print("  [Models]  No models yet — visit http://localhost:8000/hf to download")
        except Exception as e:
            print(f"  [Models]  Warning: {e}")
        return True
    except ImportError:
        print("  [Backend] llama-cpp-python not found — trying Ollama fallback...")
        print("            Run `python setup_v7.py` to install the optimised backend")

    # 2. Ollama fallback
    def _ping_ollama() -> bool:
        try:
            r = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    if _ping_ollama():
        print("  [Backend] Ollama fallback ✓")
        return True

    print("  [Backend] Ollama not detected — attempting to start...")
    try:
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["ollama","serve"], creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        elif system == "Darwin":
            try:
                subprocess.Popen(["open","-a","Ollama"])
            except Exception:
                subprocess.Popen(["ollama","serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["ollama","serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("  [Backend] Waiting for Ollama", end="", flush=True)
        for i in range(8):
            time.sleep(1)
            print(".", end="", flush=True)
            if _ping_ollama():
                print(" ✓"); return True
        print()
        print("  [Backend] Warning: no local backend running.")
        print("            Cloud models still available if an API key is set.")
        return False
    except FileNotFoundError:
        print("  [Backend] Ollama not installed.")
        print("            Run `python setup_v7.py` to set up llama-cpp-python.")
        return False


def load_admin_preferences(agent):
    """Restore all saved preferences from DB into live config + env vars."""
    import os
    try:
        # ── Agent identity ──────────────────────────────────────
        name = agent.memory.structured.get_preference("agent_name")
        if name: cfg.AGENT_NAME = name
        gender = agent.memory.structured.get_preference("agent_voice_gender")
        if gender: cfg.AGENT_VOICE_GENDER = gender
        voice = agent.memory.structured.get_preference("agent_voice_name")
        if voice: cfg.AGENT_VOICE_NAME = voice

        # ── Cloud API keys — critical for cloud_available = True ─
        or_key = agent.memory.structured.get_preference("openrouter_key")
        if or_key and or_key.strip():
            cfg.OPENROUTER_API_KEY = or_key.strip()
            os.environ["OPENROUTER_API_KEY"] = or_key.strip()
            print(f"  [Config] OpenRouter API key loaded from DB (***{or_key.strip()[-4:]})")

        or_base = agent.memory.structured.get_preference("openrouter_base")
        if or_base and or_base.strip():
            cfg.OPENROUTER_BASE = or_base.strip()
            os.environ["OPENROUTER_BASE"] = or_base.strip()

        sb_url = agent.memory.structured.get_preference("supabase_url")
        if sb_url and sb_url.strip():
            cfg.SUPABASE_URL = sb_url.strip()
            os.environ["SUPABASE_URL"] = sb_url.strip()

        sb_key = agent.memory.structured.get_preference("supabase_key")
        if sb_key and sb_key.strip():
            cfg.SUPABASE_KEY = sb_key.strip()
            os.environ["SUPABASE_KEY"] = sb_key.strip()

        # Force the router to re-check with the newly loaded keys
        agent.router.refresh()
        if or_key and or_key.strip():
            print(f"  [Config] OpenRouter key active — cloud models available.")
        else:
            print(f"  [Config] No OpenRouter key found — running in local-only mode.")
            print(f"           Add a key at https://openrouter.ai/keys then paste in Settings → Providers.")
    except Exception as e:
        print(f"  [Config] Warning loading preferences: {e}")


def start_dashboard(agent, monitor, sync):
    import uvicorn
    from dashboard.app import app, set_agent
    set_agent(agent, monitor, sync)
    uvicorn.run(app, host=cfg.DASHBOARD_HOST, port=cfg.DASHBOARD_PORT, log_level="warning")


def main():
    banner()

    for d in [cfg.DOCUMENTS_DIR, cfg.UPLOADS_DIR, cfg.GENERATED_DIR,
              cfg.LOG_DIR, getattr(cfg, "HF_MODELS_DIR", str(Path(cfg.DATA_DIR) / "hf_models"))]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.ingest:
        from ingestion.ingest import ingest_directory
        print(f"Ingesting from: {cfg.DOCUMENTS_DIR}")
        r = ingest_directory()
        print(f"Done: {r['files_processed']} files, {r['total_chunks']} chunks.")
        return

    # Auto-start Ollama
    ensure_local_backend()

    print("Starting agent...")
    try:
        from agent.orchestrator import PersonalAgent
        agent = PersonalAgent()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print(f"  2. Pull a model:  ollama pull {cfg.OLLAMA_MODEL_FAST}")
        print("  3. OR set OPENROUTER_API_KEY in your environment")
        sys.exit(1)

    load_admin_preferences(agent)

    monitor = None
    if not args.no_monitor and cfg.MONITOR_ENABLED:
        from agent.system_monitor import SystemMonitor
        monitor = SystemMonitor(agent.memory)
        monitor.start()
        print("[Monitor] System monitoring started.")

    sync = None
    if not args.no_sync:
        from sync.manager import SyncManager
        sync = SyncManager(agent.memory)
        sync.start_background_sync()

    from ingestion.ingest import ingest_directory
    print("Indexing documents...")
    r = ingest_directory(agent.memory.vector)
    if r["files_processed"]:
        print(f"  Indexed {r['files_processed']} files, {r['total_chunks']} chunks.")

    if not args.no_scheduler:
        from scheduler.jobs import create_scheduler, set_agent as sa
        sa(agent)
        sched = create_scheduler()
        sched.start()
        print("[Scheduler] Background jobs running.")

    if not args.no_dashboard:
        t = threading.Thread(target=start_dashboard, args=(agent, monitor, sync), daemon=True)
        t.start()
        print()
        print(f"  ┌─────────────────────────────────────────────┐")
        print(f"  │  Dashboard → http://localhost:{cfg.DASHBOARD_PORT}          │")
        print(f"  │  Press Ctrl+C to stop                       │")
        print(f"  └─────────────────────────────────────────────┘")
        print()
        print("  Models:", agent.stats()["models"])
        print("  Memory:", agent.stats()["memory"])
        print()

    print(f"  Terminal chat ready (or use the dashboard above)")
    print("  Commands: 'tasks' 'stats' 'quit'\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not user: continue
        if user.lower() in ("quit","exit"): break
        if user.lower() == "stats":
            import json; print(json.dumps(agent.stats(), indent=2)); continue
        if user.lower() == "tasks":
            for t in agent.get_tasks():
                print(f"  [{t['status']}] {t['title']}")
            continue
        r = agent.chat(user, verbose=True)
        print(f"\n{cfg.AGENT_NAME}: {r['response']}\n")

    if monitor: monitor.stop()
    if sync:    sync.stop()


if __name__ == "__main__":
    main()
