# Aria Personal Agent v7 — HuggingFace + llama-cpp Edition

> LM Studio-style local model manager, built into your own agent.
> Download any GGUF model from HuggingFace. Run it directly — no Ollama needed.
> System intelligence auto-detects your GPU/CPU and allocates resources optimally.

---

## What's New in v7

| Feature | v6 (Ollama) | v7 (llama-cpp + HF) |
|---|---|---|
| Backend | Ollama (separate process) | llama-cpp-python (direct, fast) |
| Model source | Ollama Hub | HuggingFace Hub (any GGUF) |
| Uncensored models | Limited | Any model, any fine-tune |
| GPU allocation | Automatic (Ollama) | Intelligent VRAM-aware layer splitting |
| System analysis | None | CPU/RAM/VRAM/CUDA/Metal detection |
| Quantisation advice | None | Recommends optimal quant per hardware |
| Model management UI | Settings tab | Dedicated HF Manager page at /hf |
| Download tracking | None | Live progress with speed indicator |
| Import local .gguf | No | Register any file already on disk |
| Ollama fallback | Primary | Auto-fallback if llama-cpp unavailable |

---

## Quick Start

### 1. One-time Setup

    python setup_v7.py

This script auto-detects NVIDIA CUDA / Apple Metal / AMD ROCm / CPU-only
and installs llama-cpp-python with the correct GPU compilation flags.

### 2. Start Aria

    python main.py

Or double-click run_windows.bat (Windows) or run_mac.command (macOS).

### 3. Download Your First Model

Open http://localhost:8000/hf and:
1. Check the Hardware tab to see your VRAM, RAM, and recommended quant
2. Go to Discover and browse or search HuggingFace GGUF models
3. Click any model, select a quantisation, and click the download button
4. Go to My Models, click the model, and press Load Model
5. Return to the main dashboard and start chatting

---

## Architecture

    main.py
      ensure_local_backend()
        llama-cpp-python  (preferred — direct, fast, no subprocess)
          LocalModelManager
            SystemAnalyser        GPU/CPU/RAM detection
            HFModelSearcher       HuggingFace Hub API + curated list
            ModelDownloadManager  Background downloads with progress
            ModelRegistry         SQLite persistence
            LlamaCppEngine        Thread-safe inference engine
        Ollama HTTP API  (automatic fallback if llama-cpp not installed)

---

## Supported Hardware

| Hardware | Backend | Notes |
|---|---|---|
| NVIDIA GPU | CUDA | Best performance, setup_v7.py handles install |
| Apple Silicon | Metal | Unified memory, fast on M1/M2/M3/M4 |
| AMD GPU | ROCm | Linux only |
| CPU-only | Default | Works everywhere, slower |

---

## Recommended Models by Hardware

| Available RAM/VRAM | Recommended Quant | Example Models |
|---|---|---|
| 4 GB | Q4_0 | Llama 3.2 3B, Phi 3.5 Mini |
| 6 GB | Q4_K_M | Mistral 7B, Llama 3.1 8B, Qwen2.5 7B |
| 10 GB | Q5_K_M | Qwen2.5 14B, Phi-4 |
| 16 GB | Q6_K | Gemma2 9B, Qwen2.5 14B |
| 24 GB+ | Q8_0 | Llama 3.1 70B (Q3), Qwen2.5 72B |

The HF Manager shows a personalised recommendation based on your exact system.

---

## New API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /hf | GET | HF Model Manager UI |
| /api/hf/system | GET | Hardware capabilities and recommendations |
| /api/hf/models | GET | List installed models |
| /api/hf/search | GET | Search HuggingFace Hub |
| /api/hf/repo/files | GET | List GGUF files in a repo with sizes |
| /api/hf/download | POST | Start background download |
| /api/hf/downloads | GET | All download tasks and progress |
| /api/hf/load | POST | Load model into inference engine |
| /api/hf/unload | POST | Unload model and free memory |
| /api/hf/settings | POST | Update gpu_layers and context length |
| /api/hf/remove | POST | Remove model from registry |
| /api/hf/import | POST | Register an existing .gguf file |
| /api/hf/recommend | GET | Get recommendation for a model size |

---

## Troubleshooting

llama-cpp-python fails to import:
    python setup_v7.py

Force CUDA install manually:
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

Model loads slowly (mode: cpu_only):
- Increase gpu_layers in HF Manager > My Models > click model > Settings
- Try a smaller quantisation (Q4_K_M instead of Q5_K_M)

Out of memory during inference:
- Reduce context_len from 4096 to 2048
- Use a more compressed quant (Q3_K_M)

HuggingFace download fails for gated models:
- Get a token at https://huggingface.co/settings/tokens
- Enter it in HF Manager > Import Local > HF Token section

---

## Environment Variables

    HF_TOKEN=hf_xxx           Optional, for gated HuggingFace models
    LOCAL_BACKEND=auto        auto | llama_cpp | ollama
    OPENROUTER_API_KEY=sk-..  Cloud models (unchanged from v6)
