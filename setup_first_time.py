#!/usr/bin/env python3
"""
Aria Setup — Run this once before first launch
python setup_first_time.py
"""

import subprocess, sys, platform, os
from pathlib import Path

OS = platform.system()

def check(label, ok, hint=""):
    """Checks a condition and prints a message based on whether it's true or false. Optionally includes a hint for troubleshooting."""
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}")
    if not ok and hint:
        print(f"      → {hint}")
    return ok


def run(cmd, **kw):
    """Executes a shell command and captures its output, handling errors."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, **kw)
    except Exception:
        return None


print()
print("=" * 52)
print("  🌿  Aria Personal Agent — First-Time Setup")
print("=" * 52)
print()

# ── Python ────────────────────────────────────────────────
pv = sys.version_info
check(f"Python {pv.major}.{pv.minor}", pv >= (3, 10),
      "Install Python 3.10+ from https://python.org")

# ── Pip packages ─────────────────────────────────────────
print("\n📦 Installing Python packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
                "--quiet", "--disable-pip-version-check"])
print("   Done.")

# ── Ollama ───────────────────────────────────────────────
print("\n🤖 Checking Ollama (local AI)...")
import requests
try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=3)
    ollama_ok = resp.status_code == 200
except Exception:
    ollama_ok = False

check("Ollama running", ollama_ok,
      "Install from https://ollama.com then run: ollama serve")

if ollama_ok:
    models = [m["name"] for m in resp.json().get("models", [])]
    has_model = any("llama3" in m or "mistral" in m for m in models)
    check(f"Model available ({', '.join(models[:3]) if models else 'none'})",
          has_model, "Run: ollama pull llama3.2")
    if not has_model:
        print("\n   Pulling llama3.2 (≈2GB, one time)...")
        subprocess.run(["ollama", "pull", "llama3.2"])
        print("   Done!")

# ── Directories ──────────────────────────────────────────
print("\n📁 Creating directories...")
for d in ["data/documents", "data/uploads", "data/generated", "data/logs/notes",
          "data/chroma_db"]:
    Path(d).mkdir(parents=True, exist_ok=True)
print("   Done.")

# ── Make launchers executable (Mac/Linux) ─────────────────
if OS != "Windows":
    for f in ["run_mac.command"]:
        fp = Path(f)
        if fp.exists():
            fp.chmod(0o755)
    print("\n🚀 Mac launcher made executable.")

# ── Summary ───────────────────────────────────────────────
print()
print("=" * 52)
print("  Setup complete!")
print()
if OS == "Darwin":
    print("  To start Aria:")
    print("    Double-click: run_mac.command")
    print("    Or in terminal: python main.py")
elif OS == "Windows":
    print("  To start Aria:")
    print("    Double-click: run_windows.bat")
    print("    Or in terminal: python main.py")
print()
print("  Dashboard:  http://localhost:8000")
print()
print("  Optional — for cloud models (free):")
print("    Get OpenRouter key: https://openrouter.ai")
print("    Get Groq key:       https://console.groq.com")
print("    Set in Settings tab or as environment variables:")
print("    export OPENROUTER_API_KEY=sk-or-...")
print()
print("  Optional — for cross-device sync (free):")
print("    Create project at: https://supabase.com")
print("    Set SUPABASE_URL and SUPABASE_KEY in environment")
print("=" * 52)
print()
