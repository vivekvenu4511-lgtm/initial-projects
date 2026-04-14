# ============================================================
#  Local Model Manager v7 — HuggingFace + llama-cpp-python
#  Replaces Ollama dependency with direct model management.
#  Features:
#    • System capability detection (RAM / VRAM / CPU cores)
#    • HuggingFace Hub model search & GGUF download
#    • Intelligent GPU-layer allocation per model size
#    • Quantization advisor (recommends best quant for hardware)
#    • Thread-safe inference with streaming support
#    • Model registry persisted in SQLite
# ============================================================

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import re
import sqlite3
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger("aria.local_model_manager")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg

# ─── Optional heavy imports — fail gracefully ─────────────────
try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not installed. Run: pip install huggingface-hub")

try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Run: pip install llama-cpp-python")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ─── Data classes ─────────────────────────────────────────────

@dataclass
class SystemCapabilities:
    """Snapshot of what this machine can actually do."""
    platform:        str = ""
    cpu_cores:       int = 1
    cpu_threads:     int = 1
    cpu_name:        str = ""
    ram_total_gb:    float = 0.0
    ram_available_gb: float = 0.0
    gpu_available:   bool = False
    gpu_name:        str = ""
    vram_total_gb:   float = 0.0
    vram_free_gb:    float = 0.0
    gpu_layers_max:  int = 0   # how many transformer layers can fit in VRAM
    metal_available: bool = False   # Apple Silicon MPS
    cuda_available:  bool = False
    recommended_quant: str = "Q4_K_M"
    max_model_size_gb: float = 0.0
    inference_threads: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelRecord:
    """A locally available model tracked in the registry."""
    id:           str           # unique slug  e.g. "TheBloke/Mistral-7B-v0.1-GGUF"
    name:         str
    repo_id:      str           # HuggingFace repo id
    filename:     str           # GGUF filename on disk
    path:         str           # absolute path to .gguf file
    size_gb:      float = 0.0
    quant:        str  = ""     # e.g. Q4_K_M
    param_size:   str  = ""     # e.g. 7B
    family:       str  = ""     # llama / mistral / qwen / phi …
    context_len:  int  = 4096
    gpu_layers:   int  = -1     # -1 = auto
    status:       str  = "ready"  # ready | loading | error
    added_at:     str  = ""
    last_used:    str  = ""
    tags:         list = field(default_factory=list)
    description:  str  = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["provider"] = "local_hf"
        d["size"] = int(self.size_gb * 1024**3)
        d["parameter_size"] = self.param_size
        d["modified_at"] = self.last_used or self.added_at
        return d


# ─── System Analyser ──────────────────────────────────────────

class SystemAnalyser:
    """Detects hardware and recommends optimal inference settings."""

    @staticmethod
    def analyse() -> SystemCapabilities:
        caps = SystemCapabilities()
        caps.platform = platform.system()

        # ── CPU ────────────────────────────────────────────────
        caps.cpu_threads = os.cpu_count() or 1
        caps.cpu_cores   = max(1, caps.cpu_threads // 2)

        try:
            import cpuinfo  # optional: pip install py-cpuinfo
            info = cpuinfo.get_cpu_info()
            caps.cpu_name = info.get("brand_raw", "")
        except Exception:
            caps.cpu_name = platform.processor() or "Unknown CPU"

        # ── RAM ────────────────────────────────────────────────
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            caps.ram_total_gb     = round(vm.total / 1024**3, 2)
            caps.ram_available_gb = round(vm.available / 1024**3, 2)
        else:
            # Fallback — read /proc/meminfo on Linux
            try:
                with open("/proc/meminfo") as f:
                    lines = f.read()
                total = int(re.search(r"MemTotal:\s+(\d+)", lines).group(1))
                free  = int(re.search(r"MemAvailable:\s+(\d+)", lines).group(1))
                caps.ram_total_gb     = round(total / 1024**2, 2)
                caps.ram_available_gb = round(free  / 1024**2, 2)
            except Exception:
                caps.ram_total_gb = caps.ram_available_gb = 4.0

        # ── GPU / VRAM ─────────────────────────────────────────
        # 1. CUDA (NVIDIA)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                caps.gpu_name        = parts[0]
                caps.vram_total_gb   = round(int(parts[1]) / 1024, 2)
                caps.vram_free_gb    = round(int(parts[2]) / 1024, 2)
                caps.cuda_available  = True
                caps.gpu_available   = True
        except Exception:
            pass

        # 2. Apple Metal (MPS) — macOS
        if caps.platform == "Darwin" and not caps.gpu_available:
            try:
                import subprocess
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5
                )
                if "Metal" in result.stdout:
                    caps.metal_available = True
                    caps.gpu_available   = True
                    # Apple Unified Memory — share with system RAM
                    caps.vram_total_gb   = caps.ram_total_gb * 0.75
                    caps.vram_free_gb    = caps.ram_available_gb * 0.70
                    # Try to get GPU name
                    m = re.search(r"Chipset Model:\s+(.+)", result.stdout)
                    if m:
                        caps.gpu_name = m.group(1).strip()
            except Exception:
                pass

        # 3. AMD ROCm
        if not caps.gpu_available:
            try:
                import subprocess
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram", "--json"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    for card in data.values():
                        caps.vram_total_gb = round(
                            int(card.get("VRAM Total Memory (B)", 0)) / 1024**3, 2)
                        caps.vram_free_gb  = round(
                            int(card.get("VRAM Total Used Memory (B)", 0)) / 1024**3, 2)
                        caps.gpu_name      = "AMD GPU (ROCm)"
                        caps.gpu_available = True
                        break
            except Exception:
                pass

        # ── Derived recommendations ────────────────────────────
        caps = SystemAnalyser._compute_recommendations(caps)
        return caps

    @staticmethod
    def _compute_recommendations(caps: SystemCapabilities) -> SystemCapabilities:
        """Compute optimal quant, gpu_layers, max_model_size, threads."""
        effective_memory = caps.vram_free_gb if caps.gpu_available else caps.ram_available_gb

        # Quantisation recommendation based on available memory
        if effective_memory >= 24:
            caps.recommended_quant = "Q8_0"
            caps.max_model_size_gb = 70.0
        elif effective_memory >= 16:
            caps.recommended_quant = "Q6_K"
            caps.max_model_size_gb = 34.0
        elif effective_memory >= 10:
            caps.recommended_quant = "Q5_K_M"
            caps.max_model_size_gb = 13.0
        elif effective_memory >= 6:
            caps.recommended_quant = "Q4_K_M"
            caps.max_model_size_gb = 7.5
        elif effective_memory >= 4:
            caps.recommended_quant = "Q4_0"
            caps.max_model_size_gb = 4.5
        else:
            caps.recommended_quant = "Q3_K_M"
            caps.max_model_size_gb = 3.0

        # GPU layer estimate: rough heuristic ~128 MB per layer for 7B models
        # Actual value will depend on model architecture, but this is a safe starting point
        if caps.gpu_available and caps.vram_free_gb > 0:
            caps.gpu_layers_max = min(80, int(caps.vram_free_gb * 1024 / 130))
        else:
            caps.gpu_layers_max = 0

        # Inference threads: leave 2 cores for OS
        caps.inference_threads = max(1, min(caps.cpu_threads - 2, 12))

        return caps

    @staticmethod
    def recommend_for_model(model_size_gb: float, caps: SystemCapabilities) -> dict:
        """Recommend gpu_layers and context for a specific model given system caps."""
        vram = caps.vram_free_gb if caps.gpu_available else 0.0
        ram  = caps.ram_available_gb

        # How much of the model can we fit in VRAM?
        if vram >= model_size_gb:
            gpu_layers = -1  # all layers in GPU
            mode = "full_gpu"
        elif vram > 0:
            fraction   = vram / model_size_gb
            # Approximate: most 7B Q4 models have ~32 layers
            est_layers = int(fraction * 32)
            gpu_layers = max(1, est_layers)
            mode = "split_gpu_cpu"
        else:
            gpu_layers = 0
            mode = "cpu_only"

        # Context window — reduce if RAM is tight
        available_for_context = ram - model_size_gb
        if available_for_context > 8:
            ctx = 8192
        elif available_for_context > 4:
            ctx = 4096
        else:
            ctx = 2048

        fits = (model_size_gb <= ram * 1.1)

        return {
            "gpu_layers":   gpu_layers,
            "context_len":  ctx,
            "mode":         mode,
            "fits_in_ram":  fits,
            "warning":      "" if fits else f"Model requires ~{model_size_gb:.1f} GB but only {ram:.1f} GB RAM available."
        }


# ─── HuggingFace Model Search ─────────────────────────────────

class HFModelSearcher:
    """Searches HuggingFace Hub for GGUF models."""

    POPULAR_GGUF_REPOS = [
        # Format: (repo_id, display_name, tags, param_size, family)
        ("bartowski/Llama-3.2-3B-Instruct-GGUF",    "Llama 3.2 3B Instruct",    ["llama","general","fast"],  "3B",  "llama"),
        ("bartowski/Llama-3.2-1B-Instruct-GGUF",    "Llama 3.2 1B Instruct",    ["llama","general","ultra-fast"], "1B", "llama"),
        ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF","Llama 3.1 8B Instruct",   ["llama","general","coding"], "8B",  "llama"),
        ("bartowski/Meta-Llama-3.1-70B-Instruct-GGUF","Llama 3.1 70B Instruct", ["llama","general","reasoning"],"70B","llama"),
        ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral 7B Instruct v0.3", ["mistral","general","fast"], "7B",  "mistral"),
        ("bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF","Mixtral 8x7B MoE",        ["mistral","general","reasoning"],"47B","mistral"),
        ("bartowski/Qwen2.5-7B-Instruct-GGUF",      "Qwen 2.5 7B Instruct",     ["qwen","general","multilingual"],"7B","qwen"),
        ("bartowski/Qwen2.5-14B-Instruct-GGUF",     "Qwen 2.5 14B Instruct",    ["qwen","reasoning","coding"], "14B", "qwen"),
        ("bartowski/Qwen2.5-72B-Instruct-GGUF",     "Qwen 2.5 72B Instruct",    ["qwen","reasoning","coding"], "72B", "qwen"),
        ("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF","Qwen 2.5 Coder 7B",        ["qwen","coding","fast"],     "7B",  "qwen"),
        ("bartowski/Qwen2.5-Coder-32B-Instruct-GGUF","Qwen 2.5 Coder 32B",      ["qwen","coding","reasoning"],"32B", "qwen"),
        ("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF","DeepSeek R1 Distill 7B", ["deepseek","reasoning","math"],"7B","deepseek"),
        ("bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF","DeepSeek R1 Distill 8B",["deepseek","reasoning","math"],"8B","deepseek"),
        ("bartowski/phi-4-GGUF",                    "Phi-4 14B",                ["phi","reasoning","coding"], "14B", "phi"),
        ("bartowski/Phi-3.5-mini-instruct-GGUF",    "Phi 3.5 Mini 3.8B",        ["phi","fast","edge"],        "3.8B","phi"),
        ("bartowski/gemma-2-9b-it-GGUF",            "Gemma 2 9B Instruct",      ["gemma","general","google"], "9B",  "gemma"),
        ("bartowski/gemma-2-27b-it-GGUF",           "Gemma 2 27B Instruct",     ["gemma","general","google"], "27B", "gemma"),
        # Uncensored / fine-tuned
        ("bartowski/dolphin-2.9.4-gemma2-2b-GGUF",  "Dolphin 2.9 Gemma2 2B",   ["dolphin","uncensored","fast"],"2B","gemma"),
        ("bartowski/dolphin-2.9.3-mistral-7B-GGUF", "Dolphin Mistral 7B",       ["dolphin","uncensored"],     "7B",  "mistral"),
        ("bartowski/dolphin-mixtral-8x7b-GGUF",     "Dolphin Mixtral 8x7B",     ["dolphin","uncensored","large"],"47B","mistral"),
        ("bartowski/Hermes-3-Llama-3.1-8B-GGUF",    "Hermes 3 Llama 3.1 8B",   ["hermes","creative","roleplay"],"8B","llama"),
        ("bartowski/Hermes-3-Llama-3.1-70B-GGUF",   "Hermes 3 Llama 3.1 70B",  ["hermes","creative","large"], "70B", "llama"),
        ("bartowski/WizardCoder-Python-34B-V1.0-GGUF","WizardCoder Python 34B", ["coding","wizard"],          "34B", "codellama"),
        ("TheBloke/CodeLlama-7B-Instruct-GGUF",     "Code Llama 7B Instruct",   ["coding","llama","meta"],    "7B",  "codellama"),
        ("TheBloke/CodeLlama-34B-Instruct-GGUF",    "Code Llama 34B Instruct",  ["coding","llama","large"],   "34B", "codellama"),
        ("bartowski/Yi-1.5-34B-Chat-GGUF",          "Yi 1.5 34B Chat",          ["yi","multilingual","large"],"34B", "yi"),
        ("bartowski/internlm2_5-20b-chat-GGUF",     "InternLM 2.5 20B Chat",    ["internlm","general","multilingual"],"20B","internlm"),
    ]

    def __init__(self):
        self._api = HfApi() if HF_AVAILABLE else None

    def search_gguf(self, query: str = "", limit: int = 20,
                    filter_tags: list[str] | None = None) -> list[dict]:
        """Search HuggingFace for GGUF models. Falls back to curated list."""
        results = []
        if HF_AVAILABLE and self._api:
            try:
                models = self._api.list_models(
                    search=query or "gguf",
                    filter="gguf",
                    limit=limit,
                    sort="downloads",
                    direction=-1,
                )
                for m in models:
                    tags = list(m.tags or [])
                    if filter_tags and not any(t in tags for t in filter_tags):
                        continue
                    results.append({
                        "repo_id":     m.id,
                        "name":        m.id.split("/")[-1].replace("-GGUF","").replace("-gguf",""),
                        "downloads":   getattr(m, "downloads", 0),
                        "likes":       getattr(m, "likes", 0),
                        "tags":        tags,
                        "source":      "hub_search",
                        "family":      self._guess_family(m.id),
                        "param_size":  self._guess_param_size(m.id),
                    })
                return results
            except Exception as e:
                logger.warning(f"HF API search failed: {e}. Returning curated list.")

        # Fallback — curated popular repos
        q = (query or "").lower()
        for repo_id, name, tags, param, family in self.POPULAR_GGUF_REPOS:
            if q and q not in repo_id.lower() and q not in name.lower():
                continue
            if filter_tags and not any(t in tags for t in filter_tags):
                continue
            results.append({
                "repo_id":    repo_id,
                "name":       name,
                "downloads":  0,
                "likes":      0,
                "tags":       tags,
                "source":     "curated",
                "family":     family,
                "param_size": param,
            })
        return results[:limit]

    def list_repo_gguf_files(self, repo_id: str) -> list[dict]:
        """List all GGUF files in a HF repo with size information."""
        if not HF_AVAILABLE:
            return []
        try:
            api   = HfApi()
            files = api.list_repo_files(repo_id)
            result = []
            for f in files:
                if not f.endswith(".gguf"):
                    continue
                quant = self._guess_quant(f)
                result.append({
                    "filename":  f,
                    "quant":     quant,
                    "quality":   self._quant_quality(quant),
                    "recommended": False,
                })
            # Mark recommended
            if result:
                best = self._pick_best_quant(result)
                for r in result:
                    r["recommended"] = (r["filename"] == best)
            return result
        except Exception as e:
            logger.error(f"list_repo_gguf_files({repo_id}): {e}")
            return []

    def list_repo_gguf_files_with_sizes(self, repo_id: str,
                                         caps: SystemCapabilities | None = None) -> list[dict]:
        """Same as above but with metadata including file sizes."""
        if not HF_AVAILABLE:
            return []
        try:
            api   = HfApi()
            files_info = list(api.list_repo_tree(repo_id, recursive=True))
            result = []
            for item in files_info:
                if not hasattr(item, "path") or not item.path.endswith(".gguf"):
                    continue
                size_bytes = getattr(item, "size", 0) or 0
                size_gb    = round(size_bytes / 1024**3, 2)
                quant      = self._guess_quant(item.path)
                rec        = self._recommend_for_caps(size_gb, quant, caps) if caps else {}
                result.append({
                    "filename":    item.path,
                    "quant":       quant,
                    "size_bytes":  size_bytes,
                    "size_gb":     size_gb,
                    "quality":     self._quant_quality(quant),
                    "recommended": False,
                    "fits":        rec.get("fits", True),
                    "note":        rec.get("note", ""),
                })
            # Mark recommended quant
            if result:
                best = self._pick_best_quant(result)
                for r in result:
                    r["recommended"] = (r["filename"] == best)
            return sorted(result, key=lambda x: x.get("size_bytes", 0))
        except Exception as e:
            logger.error(f"list_repo_gguf_files_with_sizes({repo_id}): {e}")
            return self.list_repo_gguf_files(repo_id)

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _guess_family(repo_id: str) -> str:
        r = repo_id.lower()
        for kw in ["llama","mistral","qwen","phi","gemma","deepseek","falcon",
                   "codellama","vicuna","dolphin","hermes","wizard","yi","internlm"]:
            if kw in r: return kw
        return "unknown"

    @staticmethod
    def _guess_param_size(repo_id: str) -> str:
        m = re.search(r"(\d+\.?\d*)[Bb]", repo_id.replace("-","").replace("_",""))
        return f"{m.group(1)}B" if m else ""

    @staticmethod
    def _guess_quant(filename: str) -> str:
        for q in ["Q8_0","Q6_K","Q5_K_M","Q5_K_S","Q5_0","Q4_K_M","Q4_K_S",
                  "Q4_0","Q3_K_M","Q3_K_S","Q3_K_L","Q2_K","IQ4_XS","IQ3_XXS","f16","f32"]:
            if q.lower() in filename.lower():
                return q
        return "unknown"

    @staticmethod
    def _quant_quality(quant: str) -> str:
        order = {"f32":9,"f16":8,"Q8_0":7,"Q6_K":6,"Q5_K_M":5,"Q5_K_S":5,"Q5_0":4,
                 "Q4_K_M":4,"Q4_K_S":3,"Q4_0":3,"Q3_K_M":2,"Q3_K_S":2,"Q3_K_L":2,
                 "Q2_K":1,"IQ4_XS":4,"IQ3_XXS":2}
        v = order.get(quant, 0)
        if v >= 7: return "lossless"
        if v >= 5: return "high"
        if v >= 4: return "balanced"
        if v >= 2: return "compressed"
        return "aggressive"

    @staticmethod
    def _pick_best_quant(files: list[dict]) -> str:
        """Pick Q4_K_M as universal default, fallback to whatever is closest."""
        preference = ["Q4_K_M","Q5_K_M","Q4_K_S","Q4_0","Q5_K_S","Q6_K","Q3_K_M","Q8_0","f16"]
        for pref in preference:
            for f in files:
                if f["quant"] == pref:
                    return f["filename"]
        return files[0]["filename"] if files else ""

    @staticmethod
    def _recommend_for_caps(size_gb: float, quant: str, caps: SystemCapabilities) -> dict:
        avail = caps.vram_free_gb if caps.gpu_available else caps.ram_available_gb
        fits  = size_gb <= avail * 0.95
        note  = "" if fits else f"Needs ~{size_gb:.1f}GB, you have {avail:.1f}GB free"
        return {"fits": fits, "note": note}


# ─── Download Manager ─────────────────────────────────────────

class ModelDownloadManager:
    """Handles HF downloads with progress tracking."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, dict] = {}  # task_id -> progress info
        self._lock = threading.Lock()

    def start_download(self, repo_id: str, filename: str,
                       hf_token: str = "", task_id: str | None = None) -> str:
        """Launch background download; returns task_id."""
        import uuid
        task_id = task_id or str(uuid.uuid4())[:8]
        with self._lock:
            self._active[task_id] = {
                "task_id":   task_id,
                "repo_id":   repo_id,
                "filename":  filename,
                "status":    "queued",
                "progress":  0,
                "size_bytes": 0,
                "downloaded": 0,
                "speed_mbps": 0,
                "error":     "",
                "local_path": "",
                "started_at": time.time(),
            }
        t = threading.Thread(
            target=self._download_worker,
            args=(task_id, repo_id, filename, hf_token),
            daemon=True
        )
        t.start()
        return task_id

    def _download_worker(self, task_id: str, repo_id: str,
                         filename: str, hf_token: str):
        if not HF_AVAILABLE:
            self._set_error(task_id, "huggingface_hub not installed")
            return
        try:
            self._update(task_id, status="downloading")
            dest_dir = self.models_dir / repo_id.replace("/", "__")
            dest_dir.mkdir(parents=True, exist_ok=True)

            kwargs: dict[str, Any] = dict(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest_dir),
            )
            if hf_token:
                kwargs["token"] = hf_token

            # hf_hub_download does not expose byte-level progress easily,
            # so we poll the file size in a side-thread while downloading.
            dest_path = dest_dir / filename.split("/")[-1]
            poll_stop = threading.Event()

            def poll_progress():
                while not poll_stop.is_set():
                    if dest_path.exists():
                        downloaded = dest_path.stat().st_size
                        elapsed    = max(0.1, time.time() - self._active[task_id]["started_at"])
                        speed_mbs  = round(downloaded / elapsed / 1024**2, 2)
                        with self._lock:
                            d = self._active.get(task_id, {})
                            total = d.get("size_bytes", 0) or 1
                            self._active[task_id].update({
                                "downloaded":  downloaded,
                                "progress":    min(99, int(downloaded / total * 100)) if total else 0,
                                "speed_mbps":  speed_mbs,
                            })
                    time.sleep(1)

            poll_t = threading.Thread(target=poll_progress, daemon=True)
            poll_t.start()

            local_path = hf_hub_download(**kwargs)
            poll_stop.set()
            poll_t.join(timeout=2)

            self._update(task_id,
                         status="done",
                         progress=100,
                         local_path=local_path,
                         downloaded=Path(local_path).stat().st_size if Path(local_path).exists() else 0)

        except Exception as e:
            self._set_error(task_id, str(e))

    def _update(self, task_id: str, **kwargs):
        with self._lock:
            if task_id in self._active:
                self._active[task_id].update(kwargs)

    def _set_error(self, task_id: str, error: str):
        self._update(task_id, status="error", error=error)

    def get_progress(self, task_id: str) -> dict:
        with self._lock:
            return dict(self._active.get(task_id, {}))

    def list_active(self) -> list[dict]:
        with self._lock:
            return [dict(v) for v in self._active.values()]

    def cancel_not_supported(self) -> str:
        return "Cancellation requires restarting the download. Stop the server to abort."


# ─── Model Registry (SQLite) ──────────────────────────────────

class ModelRegistry:
    """Persists model records in the agent SQLite database."""

    TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS local_hf_models (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        repo_id      TEXT NOT NULL,
        filename     TEXT NOT NULL,
        path         TEXT NOT NULL,
        size_gb      REAL DEFAULT 0,
        quant        TEXT DEFAULT '',
        param_size   TEXT DEFAULT '',
        family       TEXT DEFAULT '',
        context_len  INTEGER DEFAULT 4096,
        gpu_layers   INTEGER DEFAULT -1,
        status       TEXT DEFAULT 'ready',
        added_at     TEXT DEFAULT (datetime('now')),
        last_used    TEXT DEFAULT '',
        tags         TEXT DEFAULT '[]',
        description  TEXT DEFAULT ''
    )
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()

    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=15)
        c.row_factory = sqlite3.Row
        # WAL mode: allows concurrent readers + one writer (no full lock on reads)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    def _ensure_table(self):
        with self._conn() as c:
            c.executescript(self.TABLE_DDL)

    def add_or_update(self, rec: ModelRecord):
        d = asdict(rec)
        d["tags"] = json.dumps(d.get("tags", []))
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO local_hf_models
                (id, name, repo_id, filename, path, size_gb, quant, param_size,
                 family, context_len, gpu_layers, status, added_at, last_used, tags, description)
                VALUES (:id,:name,:repo_id,:filename,:path,:size_gb,:quant,:param_size,
                        :family,:context_len,:gpu_layers,:status,:added_at,:last_used,:tags,:description)
            """, d)

    def list_all(self) -> list[ModelRecord]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM local_hf_models ORDER BY added_at DESC").fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, model_id: str) -> ModelRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM local_hf_models WHERE id=?", (model_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def remove(self, model_id: str) -> bool:
        with self._conn() as c:
            c.execute("DELETE FROM local_hf_models WHERE id=?", (model_id,))
        return True

    def touch(self, model_id: str):
        from datetime import datetime
        with self._conn() as c:
            c.execute("UPDATE local_hf_models SET last_used=? WHERE id=?",
                      (datetime.now().isoformat(), model_id))

    def set_status(self, model_id: str, status: str):
        with self._conn() as c:
            c.execute("UPDATE local_hf_models SET status=? WHERE id=?", (status, model_id))

    @staticmethod
    def _row_to_record(row) -> ModelRecord:
        d = dict(row)
        try:
            d["tags"] = json.loads(d.get("tags") or "[]")
        except Exception:
            d["tags"] = []
        return ModelRecord(**{k: v for k, v in d.items() if k in ModelRecord.__dataclass_fields__})


# ─── Inference Engine (llama-cpp-python) ──────────────────────

class LlamaCppEngine:
    """Thread-safe llama.cpp inference engine with model hot-swap."""

    def __init__(self):
        self._llm: Optional[Any] = None   # Llama instance
        self._current_model_id: str = ""
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def current_model(self) -> str:
        return self._current_model_id

    def load(self, rec: ModelRecord, caps: SystemCapabilities) -> dict:
        """Load a model. Unloads previous model first."""
        if not LLAMA_CPP_AVAILABLE:
            return {"ok": False, "error": "llama-cpp-python not installed. Run: pip install llama-cpp-python"}

        with self._load_lock:
            # Already loaded?
            if self._current_model_id == rec.id and self._llm is not None:
                return {"ok": True, "model": rec.id, "message": "Already loaded"}

            # Unload previous
            self._unload()

            model_path = Path(rec.path)
            if not model_path.exists():
                return {"ok": False, "error": f"Model file not found: {rec.path}"}

            # Compute gpu_layers
            model_size_gb = model_path.stat().st_size / 1024**3
            rec_params    = SystemAnalyser.recommend_for_model(model_size_gb, caps)
            gpu_layers    = rec.gpu_layers if rec.gpu_layers != -1 else rec_params["gpu_layers"]

            n_threads = caps.inference_threads
            ctx       = rec.context_len or rec_params["context_len"]

            logger.info(f"Loading {rec.name} | gpu_layers={gpu_layers} | ctx={ctx} | threads={n_threads}")

            try:
                kwargs: dict[str, Any] = dict(
                    model_path = str(model_path),
                    n_ctx      = ctx,
                    n_threads  = n_threads,
                    n_gpu_layers = gpu_layers,
                    verbose    = False,
                )
                # Apple Metal flag
                if caps.metal_available:
                    kwargs["n_gpu_layers"] = gpu_layers if gpu_layers > 0 else -1

                with self._lock:
                    self._llm              = Llama(**kwargs)
                    self._current_model_id = rec.id

                return {
                    "ok":          True,
                    "model":       rec.id,
                    "gpu_layers":  gpu_layers,
                    "ctx":         ctx,
                    "threads":     n_threads,
                    "mode":        rec_params["mode"],
                }
            except Exception as e:
                logger.error(f"Failed to load {rec.name}: {e}")
                self._unload()
                return {"ok": False, "error": str(e)}

    def _unload(self):
        if self._llm is not None:
            try:
                del self._llm
            except Exception:
                pass
            self._llm = None
            self._current_model_id = ""
            gc.collect()
            logger.info("Previous model unloaded.")

    def unload(self):
        with self._load_lock:
            self._unload()

    def generate(self, messages: list[dict], *,
                 max_tokens: int = 1800,
                 temperature: float = 0.7,
                 stop: list[str] | None = None) -> str:
        """Blocking generation from chat messages."""
        if not self._llm:
            raise RuntimeError("No model loaded. Load a model first.")

        prompt = self._format_messages(messages)

        with self._lock:
            output = self._llm(
                prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
                stop        = stop or ["<|eot_id|>","</s>","[/INST]","User:","Human:"],
                echo        = False,
            )

        if isinstance(output, dict):
            choices = output.get("choices", [{}])
            return (choices[0].get("text") or "").strip()
        return str(output).strip()

    def generate_chat(self, messages: list[dict], *,
                      max_tokens: int = 1800,
                      temperature: float = 0.7) -> str:
        """Use create_chat_completion if available (faster, cleaner)."""
        if not self._llm:
            raise RuntimeError("No model loaded.")
        with self._lock:
            try:
                resp    = self._llm.create_chat_completion(
                    messages    = messages,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
                choices = resp.get("choices", [{}])
                return (choices[0].get("message", {}).get("content") or "").strip()
            except Exception:
                # Fall back to raw generation
                return self.generate(messages, max_tokens=max_tokens, temperature=temperature)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """Format chat messages into a simple prompt string (generic fallback)."""
        parts = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)


# ─── Main Manager (public API) ────────────────────────────────

class LocalModelManager:
    """
    Public facade used by ModelRouter.
    Drop-in replacement for the Ollama integration.
    """

    def __init__(self, db_path: str | None = None, models_dir: str | None = None):
        self.db_path   = db_path   or cfg.SQLITE_DB
        self.models_dir = Path(models_dir or getattr(cfg, "HF_MODELS_DIR",
                                                      str(Path(cfg.DATA_DIR) / "hf_models")))
        self.registry  = ModelRegistry(self.db_path)
        self.searcher  = HFModelSearcher()
        self.downloader = ModelDownloadManager(str(self.models_dir))
        self.engine    = LlamaCppEngine()
        self._caps: SystemCapabilities | None = None
        self._caps_lock = threading.Lock()
        self._disk_scanned = False  # lazy — scan on first list_local_models()

    # ── system caps ───────────────────────────────────────────

    def get_capabilities(self, refresh: bool = False) -> SystemCapabilities:
        with self._caps_lock:
            if self._caps is None or refresh:
                self._caps = SystemAnalyser.analyse()
            return self._caps

    # ── model listing ─────────────────────────────────────────

    def list_local_models(self) -> list[dict]:
        """Return all registered local HF models as dicts (ModelRouter-compatible)."""
        if not self._disk_scanned:
            self._disk_scanned = True
            try:
                # Run in background thread so it doesn't block the response
                import threading as _t
                _t.Thread(target=self._scan_disk_models, daemon=True).start()
            except Exception:
                pass
        records = self.registry.list_all()
        return [r.to_dict() for r in records]

    def get_model(self, model_id: str) -> dict | None:
        rec = self.registry.get(model_id)
        return rec.to_dict() if rec else None

    # ── HF search ─────────────────────────────────────────────

    def search_models(self, query: str = "", limit: int = 30,
                      tags: list[str] | None = None) -> list[dict]:
        """Search HuggingFace Hub for GGUF models."""
        results = self.searcher.search_gguf(query=query, limit=limit, filter_tags=tags)
        # Annotate which ones are already downloaded
        local_repos = {r.get("repo_id") for r in self.list_local_models()}
        for r in results:
            r["downloaded"] = r["repo_id"] in local_repos
        return results

    def get_repo_files(self, repo_id: str) -> list[dict]:
        """List GGUF files in a repo with size and quant recommendations."""
        caps = self.get_capabilities()
        return self.searcher.list_repo_gguf_files_with_sizes(repo_id, caps)

    # ── download ──────────────────────────────────────────────

    def download_model(self, repo_id: str, filename: str,
                       hf_token: str = "",
                       name: str = "", description: str = "") -> dict:
        """Start background download. Returns task_id."""
        caps    = self.get_capabilities()
        task_id = self.downloader.start_download(repo_id, filename, hf_token)

        # Register with "downloading" status so UI can track it
        model_id = f"{repo_id}/{filename}"
        from datetime import datetime
        rec = ModelRecord(
            id          = model_id,
            name        = name or f"{repo_id.split('/')[-1]} ({filename})",
            repo_id     = repo_id,
            filename    = filename,
            path        = str(self.models_dir / repo_id.replace("/","__") / filename.split("/")[-1]),
            size_gb     = 0.0,
            quant       = HFModelSearcher._guess_quant(filename),
            param_size  = HFModelSearcher._guess_param_size(repo_id),
            family      = HFModelSearcher._guess_family(repo_id),
            context_len = 4096,
            gpu_layers  = caps.gpu_layers_max,
            status      = "downloading",
            added_at    = datetime.now().isoformat(),
            description = description,
            tags        = [],
        )
        self.registry.add_or_update(rec)

        # Background watcher to flip status when done
        def _watch():
            while True:
                time.sleep(3)
                prog = self.downloader.get_progress(task_id)
                if prog.get("status") == "done":
                    local_path = prog.get("local_path", "")
                    if local_path and Path(local_path).exists():
                        size_gb = round(Path(local_path).stat().st_size / 1024**3, 2)
                        rec.path    = local_path
                        rec.size_gb = size_gb
                        rec.status  = "ready"
                        self.registry.add_or_update(rec)
                    break
                if prog.get("status") == "error":
                    self.registry.set_status(model_id, "error")
                    break

        threading.Thread(target=_watch, daemon=True).start()
        return {"task_id": task_id, "model_id": model_id, "status": "downloading"}

    def download_progress(self, task_id: str) -> dict:
        return self.downloader.get_progress(task_id)

    def list_downloads(self) -> list[dict]:
        return self.downloader.list_active()

    # ── model load/unload ─────────────────────────────────────

    def load_model(self, model_id: str) -> dict:
        """Load a model into the llama.cpp engine."""
        rec  = self.registry.get(model_id)
        if not rec:
            return {"ok": False, "error": f"Model '{model_id}' not in registry."}
        caps = self.get_capabilities()
        self.registry.set_status(model_id, "loading")
        result = self.engine.load(rec, caps)
        if result.get("ok"):
            self.registry.set_status(model_id, "ready")
            self.registry.touch(model_id)
        else:
            self.registry.set_status(model_id, "error")
        return result

    def unload_model(self):
        """Unload whatever is currently in memory."""
        self.engine.unload()

    # ── inference (used by ModelRouter) ───────────────────────

    def chat(self, messages: list[dict], model_id: str, *,
             max_tokens: int = 1800, temperature: float = 0.7,
             timeout: int | None = None) -> dict:
        """
        Main entry for chat inference.
        Auto-loads model if not already loaded.
        Returns {"content": ..., "model": ..., "latency_ms": ...}
        """
        start = time.perf_counter()

        # Auto-load if needed
        if self.engine.current_model != model_id or not self.engine.is_loaded:
            load_result = self.load_model(model_id)
            if not load_result.get("ok"):
                raise RuntimeError(f"Cannot load model: {load_result.get('error')}")

        content = self.engine.generate_chat(
            messages,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        self.registry.touch(model_id)
        return {"content": content, "model": model_id, "latency_ms": latency_ms}

    # ── ollama-compatible status dict ─────────────────────────

    def status(self) -> dict:
        """Return an Ollama-compatible status dict for the router."""
        models  = self.list_local_models()
        caps    = self.get_capabilities()
        running = LLAMA_CPP_AVAILABLE
        return {
            "running":     running,
            "connected":   running,
            "latency_ms":  0,
            "base_url":    "llama-cpp://local",
            "models":      models,
            "error":       "" if running else "llama-cpp-python not installed",
            "backend":     "llama_cpp",
            "gpu_available":   caps.gpu_available,
            "gpu_name":        caps.gpu_name,
            "vram_free_gb":    caps.vram_free_gb,
            "recommended_quant": caps.recommended_quant,
        }

    # ── remove model ──────────────────────────────────────────

    def remove_model(self, model_id: str, delete_file: bool = False) -> dict:
        """Remove from registry; optionally delete the .gguf file."""
        rec = self.registry.get(model_id)
        if not rec:
            return {"ok": False, "error": "Model not found"}
        if self.engine.current_model == model_id:
            self.engine.unload()
        if delete_file and Path(rec.path).exists():
            try:
                Path(rec.path).unlink()
            except Exception as e:
                return {"ok": False, "error": f"Could not delete file: {e}"}
        self.registry.remove(model_id)
        return {"ok": True, "model_id": model_id}

    # ── disk scan ─────────────────────────────────────────────

    def _scan_disk_models(self):
        """Find .gguf files on disk and register them if not already tracked."""
        from datetime import datetime
        registered = {r.get("path") for r in self.list_local_models()}
        for gguf in self.models_dir.rglob("*.gguf"):
            if str(gguf) in registered:
                continue
            filename = gguf.name
            # Try to infer repo_id from parent dir name (we use repo__name pattern)
            parent   = gguf.parent.name
            repo_id  = parent.replace("__", "/") if "__" in parent else f"local/{parent}"
            model_id = f"{repo_id}/{filename}"
            size_gb  = round(gguf.stat().st_size / 1024**3, 2)
            rec = ModelRecord(
                id         = model_id,
                name       = filename.replace(".gguf",""),
                repo_id    = repo_id,
                filename   = filename,
                path       = str(gguf),
                size_gb    = size_gb,
                quant      = HFModelSearcher._guess_quant(filename),
                param_size = HFModelSearcher._guess_param_size(filename),
                family     = HFModelSearcher._guess_family(filename),
                context_len = 4096,
                gpu_layers  = -1,
                status      = "ready",
                added_at    = datetime.now().isoformat(),
            )
            self.registry.add_or_update(rec)
            logger.info(f"Auto-registered: {model_id}")

    def import_local_file(self, file_path: str, name: str = "",
                           description: str = "") -> dict:
        """Register an existing .gguf file from anywhere on disk."""
        from datetime import datetime
        p = Path(file_path)
        if not p.exists() or not p.suffix == ".gguf":
            return {"ok": False, "error": "File not found or not a .gguf"}
        model_id = f"local/{p.stem}"
        size_gb  = round(p.stat().st_size / 1024**3, 2)
        rec = ModelRecord(
            id         = model_id,
            name       = name or p.stem,
            repo_id    = "local",
            filename   = p.name,
            path       = str(p),
            size_gb    = size_gb,
            quant      = HFModelSearcher._guess_quant(p.name),
            param_size = HFModelSearcher._guess_param_size(p.name),
            family     = HFModelSearcher._guess_family(p.name),
            context_len = 4096,
            gpu_layers  = -1,
            status      = "ready",
            added_at    = datetime.now().isoformat(),
            description = description,
        )
        self.registry.add_or_update(rec)
        return {"ok": True, "model_id": model_id, "size_gb": size_gb}

    def update_model_settings(self, model_id: str,
                               gpu_layers: int | None = None,
                               context_len: int | None = None,
                               name: str | None = None) -> dict:
        """Update inference parameters for a registered model."""
        rec = self.registry.get(model_id)
        if not rec:
            return {"ok": False, "error": "Model not found"}
        if gpu_layers  is not None: rec.gpu_layers  = gpu_layers
        if context_len is not None: rec.context_len = context_len
        if name        is not None: rec.name        = name
        self.registry.add_or_update(rec)
        # Unload if this is the active model (needs reload to pick up new settings)
        if self.engine.current_model == model_id:
            self.engine.unload()
        return {"ok": True}
