# ============================================================
#  Aria Personal Agent — Master Configuration v6
# ============================================================

from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLATFORM = platform.system()

AGENT_NAME       = os.getenv("AGENT_NAME", "Aria")
AGENT_VERSION    = "6.0"
DEVICE_NAME      = platform.node() or "unknown-device"
AGENT_VOICE_GENDER = os.getenv("AGENT_VOICE_GENDER", "female")
AGENT_VOICE_NAME   = os.getenv("AGENT_VOICE_NAME", "")

AGENT_PERSONA = """
You are {name}, an elite personal AI agent running on the user's system.
You are proactive, precise, and highly capable. You produce rich, well-structured answers.
When answering, use Markdown: headings, bullet lists, **bold**, tables, code blocks where appropriate.
You can read, analyze, and generate files including Excel, Word, PowerPoint, and text documents.
You remember relevant prior conversations and knowledge base documents.
When file outputs are created, tell the user how to download them from the dashboard.
Always provide thorough, illustrative responses like a top-tier AI assistant.
For news or current events, present results in a structured table format.
For data analysis, create visual summaries with statistics highlighted.
""".strip()

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_FAST  = os.getenv("OLLAMA_MODEL_FAST", "qwen2.5:latest")
OLLAMA_MODEL_CODE  = os.getenv("OLLAMA_MODEL_CODE", "qwen2.5:latest")
OLLAMA_MODEL_MAIN  = os.getenv("OLLAMA_MODEL_MAIN", "qwen2.5:latest")

OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE      = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_APP_URL   = os.getenv("OPENROUTER_APP_URL", "https://aria-agent.local")
OPENROUTER_APP_NAME  = os.getenv("OPENROUTER_APP_NAME", "Aria Personal Agent v6")
CLOUD_MODEL_BALANCED = os.getenv("CLOUD_MODEL_BALANCED", "meta-llama/llama-3.3-70b-instruct:free")
CLOUD_MODEL_CODING   = os.getenv("CLOUD_MODEL_CODING",   "qwen/qwen-2.5-coder-32b-instruct:free")
CLOUD_MODEL_REASON   = os.getenv("CLOUD_MODEL_REASON",   "deepseek/deepseek-r1:free")
CLOUD_MODEL_FAST     = os.getenv("CLOUD_MODEL_FAST",     "google/gemma-3-4b-it:free")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE    = os.getenv("GROQ_BASE", "https://api.groq.com/openai/v1")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

COMPLEXITY_CLOUD_THRESHOLD   = 3
DEFAULT_EXECUTION_MODE        = os.getenv("DEFAULT_EXECUTION_MODE", "local+cloud")
HYBRID_CLOUD_FIRST            = True   # Cloud-first in hybrid (faster)
MODEL_STATUS_CACHE_SECS       = 300  # 5 min — avoids re-pinging Ollama/OpenRouter on every chat
REQUEST_TIMEOUT_SECS          = 120   # Max total timeout
CLOUD_TIMEOUT_SECS            = 45    # Per-request cloud timeout
LOCAL_TIMEOUT_SECS            = 60    # Per-request local timeout (was 120 — too long to appear frozen)
MODEL_TEST_TIMEOUT_SECS       = 25
FILE_PREVIEW_CHARS            = 12000
INTELLIGENCE_MODE_ENABLED     = True

CHROMA_DIR      = str(DATA_DIR / "chroma_db")
SQLITE_DB       = str(DATA_DIR / "agent.db")
MEMORY_RESULTS  = 5
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64

SUPABASE_URL       = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY       = os.getenv("SUPABASE_KEY", "")
SYNC_ENABLED       = bool(SUPABASE_URL and SUPABASE_KEY)
SYNC_INTERVAL_SECS = 120

DOCUMENTS_DIR      = str(DATA_DIR / "documents")
UPLOADS_DIR        = str(DATA_DIR / "uploads")
GENERATED_DIR      = str(DATA_DIR / "generated")
LOG_DIR            = str(DATA_DIR / "logs")
HF_MODELS_DIR      = str(DATA_DIR / "hf_models")   # Where GGUF files are stored
SUPPORTED_FORMATS  = [".pdf",".txt",".md",".docx",".csv",".xlsx",".xls",".pptx",".ppt",".json"]

# ── HuggingFace / llama-cpp settings ──────────────────────────
HF_TOKEN           = os.getenv("HF_TOKEN", "")           # Optional — allows gated models
HF_DEFAULT_QUANT   = os.getenv("HF_DEFAULT_QUANT", "Q4_K_M")   # Default quantisation
LOCAL_BACKEND      = os.getenv("LOCAL_BACKEND", "auto")   # auto | llama_cpp | ollama
LLAMA_CPP_VERBOSE  = False                                # Set True for llama.cpp debug logs
LLAMA_CPP_SEED     = -1                                   # -1 = random

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8000

DAILY_SUMMARY_HOUR           = 8
TIMEZONE                     = "Asia/Dubai"
TASK_REMINDER_LOOKAHEAD_MINS = 15

MONITOR_ENABLED      = True
MONITOR_POLL_SECS    = 10
MONITOR_SUGGEST_SECS = 300

_DEFAULT_ADMIN_PW   = "aria2025"
ADMIN_PASSWORD_HASH = os.getenv(
    "ADMIN_PASSWORD_HASH",
    hashlib.sha256(_DEFAULT_ADMIN_PW.encode()).hexdigest()
)

MODEL_CAPABILITIES = {
    "qwen2.5:latest":    {"strengths":["general","reasoning","multilingual"],"best_for":"General chat, reasoning, multilingual"},
    "llama3.2":          {"strengths":["general","creative"],"best_for":"General tasks, creative writing"},
    "llama3.1:8b":       {"strengths":["general","fast"],"best_for":"Fast responses, everyday tasks"},
    "deepseek-r1:1.5b":  {"strengths":["coding","reasoning"],"best_for":"Code generation & debugging"},
    "mistral:latest":    {"strengths":["general","fast"],"best_for":"Fast general purpose"},
    "phi3:mini":         {"strengths":["fast","edge"],"best_for":"Ultra-fast lightweight tasks"},
    "gemma2:9b":         {"strengths":["general","reasoning"],"best_for":"Balanced performance"},
    "codellama:7b":      {"strengths":["coding"],"best_for":"Code tasks"},
    "meta-llama/llama-3.3-70b-instruct:free":    {"strengths":["general","reasoning","creative"],"best_for":"Best free general model"},
    "qwen/qwen-2.5-coder-32b-instruct:free":     {"strengths":["coding","reasoning"],"best_for":"Top-tier code generation"},
    "deepseek/deepseek-r1:free":                 {"strengths":["reasoning","math","analysis"],"best_for":"Deep reasoning, math, analysis"},
    "google/gemma-3-4b-it:free":                 {"strengths":["fast","general"],"best_for":"Very fast for simple queries"},
    "mistralai/mistral-7b-instruct:free":        {"strengths":["general","fast"],"best_for":"Fast general-purpose"},
    "google/gemma-3-27b-it:free":                {"strengths":["general","reasoning"],"best_for":"Larger Gemma — better quality"},
    "nousresearch/hermes-3-llama-3.1-405b:free": {"strengths":["general","creative","roleplay"],"best_for":"Creative writing, large context"},
}
