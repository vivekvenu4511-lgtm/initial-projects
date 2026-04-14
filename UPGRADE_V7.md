# Aria v7 Upgrade Notes — HuggingFace + llama-cpp

## Files Added / Changed

### New Files
- agent/local_model_manager.py   — Core HF + llama-cpp engine
- dashboard/templates/hf_manager.html  — LM Studio-style model manager UI
- setup_v7.py                    — Smart GPU-detecting installer

### Modified Files
- agent/model_router.py     — Adds llama-cpp backend, keeps Ollama as fallback
- config.py                 — Adds HF_MODELS_DIR, HF_TOKEN, LOCAL_BACKEND settings
- dashboard/app.py          — Adds all /api/hf/* endpoints
- dashboard/templates/index.html — Adds HF Manager nav link, updates backend status UI
- main.py                   — Replaces ensure_ollama() with ensure_local_backend()
- requirements.txt          — Adds huggingface-hub, notes llama-cpp variants
- run_windows.bat           — Auto-runs setup_v7.py on first launch
- run_mac.command           — Auto-runs setup_v7.py on first launch
- README.md                 — Full v7 documentation

## Database Changes
- New table: local_hf_models in data/agent.db (auto-created on first run)
- All existing data (conversations, preferences, knowledge base) preserved

## Migration from v6
No manual migration needed. The new backend:
1. On startup, scans data/hf_models/ for any existing .gguf files
2. Falls back to Ollama automatically if llama-cpp-python is not installed
3. Existing cloud model settings and API keys carry over unchanged

## Component Architecture

LocalModelManager (agent/local_model_manager.py)
├── SystemAnalyser        Detects CUDA/Metal/ROCm/CPU, RAM, VRAM
│   └── recommend_for_model()   Per-model gpu_layers + context advice
├── HFModelSearcher       HuggingFace Hub search + curated list of 25+ models
│   └── list_repo_gguf_files_with_sizes()  Size + quant + fit-check per file
├── ModelDownloadManager  hf_hub_download in background thread
│   └── Polls file size for live progress %
├── ModelRegistry         SQLite CRUD for local_hf_models table
└── LlamaCppEngine        Thread-safe Llama() wrapper
    ├── load()            Auto-computes gpu_layers from SystemAnalyser
    ├── generate_chat()   Uses create_chat_completion (faster)
    └── generate()        Raw prompt fallback

ModelRouter (agent/model_router.py)
├── local_backend_status()  Calls LocalModelManager.status() first
├── ollama_status()         Alias → local_backend_status() for compat
└── chat_local()            Calls LocalModelManager.chat(), falls back to Ollama
