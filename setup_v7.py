#!/usr/bin/env python3
# ============================================================
#  Aria Personal Agent v7 — Smart Setup Script
#  Windows: uses pre-built wheels (no compiler needed)
#  macOS/Linux: detects GPU and installs matching variant
#
#  Run once:  python setup_v7.py
#  Then run:  python main.py
# ============================================================

import os
import platform
import subprocess
import sys

PLATFORM = platform.system()
PY       = sys.executable
IS_WIN   = PLATFORM == "Windows"

BANNER = """
╔══════════════════════════════════════════════════════╗
║  Aria Personal Agent v7 — Setup                      ║
║  HuggingFace + llama-cpp-python backend installer    ║
╚══════════════════════════════════════════════════════╝
"""

# Pre-built wheel index URLs — no compiler needed
# Published by the llama-cpp-python project
HF_WHEEL_BASE = "https://abetlen.github.io/llama-cpp-python/whl"
WHEEL_INDEXES = {
    "cpu":    f"{HF_WHEEL_BASE}/cpu",
    "cu121":  f"{HF_WHEEL_BASE}/cu121",
    "cu122":  f"{HF_WHEEL_BASE}/cu122",
    "cu123":  f"{HF_WHEEL_BASE}/cu123",
    "cu124":  f"{HF_WHEEL_BASE}/cu124",
    "metal":  f"{HF_WHEEL_BASE}/metal",
}

def run(cmd, env=None, check=True, quiet=False):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    merged = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=merged,
                            capture_output=quiet, text=quiet)
    if result.returncode != 0:
        if quiet and hasattr(result, "stderr") and result.stderr:
            print(result.stderr[-800:])
        if check:
            print(f"\n  [ERROR] Command failed (exit {result.returncode})")
            return False
        # check=False: still return the actual outcome
        return False
    return True


# ── Windows Long Path fix ──────────────────────────────────────
def fix_windows_long_paths():
    """Enable Windows long path support via registry (requires admin) or pip config."""
    if not IS_WIN:
        return
    print("\n─── Windows Long Path fix ───")
    # Method 1: pip config
    run([PY, "-m", "pip", "config", "set", "global.cache-dir",
         os.path.expanduser("~/.pip_cache")], check=False, quiet=True)
    # Method 2: registry (silent, works without admin if policy allows)
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
        winreg.CloseKey(key)
        print("  ✓ Windows Long Paths enabled via registry")
    except Exception:
        print("  ⚠  Could not enable Long Paths via registry (may need admin).")
        print("     If you hit path-length errors, run this as Administrator.")
    # Method 3: use shorter pip cache path
    short_cache = "C:\\pip_cache" if IS_WIN else None
    if short_cache:
        run([PY, "-m", "pip", "config", "set", "global.cache-dir", short_cache],
            check=False, quiet=True)


# ── GPU detection ──────────────────────────────────────────────
def detect_gpu():
    """
    Detect GPU and CUDA version.
    Returns (backend_name, extra_index_url_or_None, cmake_args_dict)
    """
    print("\n─── Detecting GPU ───")

    # NVIDIA CUDA — check nvidia-smi
    for nvidia_cmd in (["nvidia-smi"], [r"C:\Windows\System32\nvidia-smi.exe"]):
        try:
            r = subprocess.run(
                nvidia_cmd + ["--query-gpu=name,memory.total",
                              "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                gpu_name = r.stdout.strip().split(",")[0].strip()
                print(f"  ✓ NVIDIA GPU: {gpu_name}")

                # Detect CUDA version
                cuda_ver = _detect_cuda_version()
                index_key = f"cu{cuda_ver.replace('.','')}" if cuda_ver else "cu122"
                if index_key not in WHEEL_INDEXES:
                    index_key = "cu122"   # safe fallback
                print(f"  ✓ CUDA version: {cuda_ver or 'unknown'} → using {index_key} wheels")
                return "cuda", WHEEL_INDEXES[index_key], {"CMAKE_ARGS": "-DGGML_CUDA=on"}
        except Exception:
            pass

    # Apple Metal
    if PLATFORM == "Darwin":
        try:
            r = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                               capture_output=True, text=True, timeout=5)
            if "Metal" in r.stdout:
                import re
                m = re.search(r"Chipset Model:\s+(.+)", r.stdout)
                name = m.group(1).strip() if m else "Apple GPU"
                print(f"  ✓ Apple Metal GPU: {name}")
                return "metal", WHEEL_INDEXES["metal"], {"CMAKE_ARGS": "-DGGML_METAL=on"}
        except Exception:
            pass

    # AMD ROCm (Linux)
    if PLATFORM == "Linux":
        try:
            r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                print("  ✓ AMD ROCm GPU detected")
                return "rocm", None, {"CMAKE_ARGS": "-DGGML_HIPBLAS=on"}
        except Exception:
            pass

    print("  ⚠  No GPU detected — CPU-only mode")
    print("     Inference will work but be slower than with a GPU.")
    return "cpu", WHEEL_INDEXES["cpu"], {}


def _detect_cuda_version():
    """Return CUDA version string like '12.2' or empty string."""
    for cmd in (["nvcc", "--version"], ["nvidia-smi"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            import re
            m = re.search(r"[Cc][Uu][Dd][Aa][^\d]*(\d+)\.(\d+)", r.stdout)
            if m:
                return f"{m.group(1)}.{m.group(2)}"
        except Exception:
            pass
    return ""


# ── llama-cpp-python install ───────────────────────────────────
def _verify_llama_installed():
    """Return True if llama_cpp can actually be imported."""
    try:
        result = subprocess.run(
            [PY, "-c", "import llama_cpp; print(llama_cpp.__version__)"],
            capture_output=True, text=True, timeout=20
        )
        return result.returncode == 0
    except Exception:
        return False


def install_llama_cpp(backend, wheel_index, cmake_args):
    print(f"\n─── Installing llama-cpp-python ({backend.upper()}) ───")

    py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    print(f"  Python version: {sys.version_info.major}.{sys.version_info.minor}")

    # Strategy 1a: pre-built wheel — force binary only (no source fallback)
    if wheel_index:
        print(f"  Trying pre-built wheels (binary-only) from:\n  {wheel_index}")
        ok = run(
            [PY, "-m", "pip", "install", "llama-cpp-python",
             "--upgrade", "--no-cache-dir",
             "--only-binary", ":all:",
             "--extra-index-url", wheel_index],
            check=False, quiet=False
        )
        if ok and _verify_llama_installed():
            print("  ✓ llama-cpp-python installed (pre-built binary wheel)")
            return True

    # Strategy 1b: allow source if binary not found, but ONLY from the wheel index
    # (avoids pulling from PyPI tarball which requires cmake)
    if wheel_index:
        print("  Binary wheel not found for your Python version.")
        print("  Trying with source allowed from wheel index (may take longer)…")
        ok = run(
            [PY, "-m", "pip", "install", "llama-cpp-python",
             "--upgrade", "--no-cache-dir",
             "--extra-index-url", wheel_index,
             "--no-build-isolation"],
            check=False, quiet=False
        )
        if ok and _verify_llama_installed():
            print("  ✓ llama-cpp-python installed")
            return True

    # Strategy 2: source build from PyPI (requires C++ compiler)
    if cmake_args and not IS_WIN:
        # Only try source build on non-Windows (compiler usually present)
        print(f"  Building from source with {cmake_args}…")
        ok = run(
            [PY, "-m", "pip", "install", "llama-cpp-python",
             "--upgrade", "--no-cache-dir", "--force-reinstall"],
            env=cmake_args, check=False
        )
        if ok and _verify_llama_installed():
            print("  ✓ llama-cpp-python installed (source build)")
            return True

    # Strategy 3: CPU pre-built fallback (if GPU install failed)
    if backend != "cpu":
        print("  Falling back to CPU-only pre-built wheel…")
        ok = run(
            [PY, "-m", "pip", "install", "llama-cpp-python",
             "--upgrade", "--no-cache-dir",
             "--only-binary", ":all:",
             "--extra-index-url", WHEEL_INDEXES["cpu"]],
            check=False
        )
        if ok and _verify_llama_installed():
            print(f"  ✓ llama-cpp-python installed (CPU-only — no {backend.upper()})")
            return True

    _print_manual_install(backend)
    return False


def _print_manual_install(backend):
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cuda_index = "cu122"
    print(f"""
  ─────────────────────────────────────────────────────
  llama-cpp-python could not be installed automatically.
  Your Python: {sys.version_info.major}.{sys.version_info.minor} ({py_tag})

  ── OPTION A: Pre-built wheels (recommended, no compiler) ──
  CPU only:
    pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

  NVIDIA GPU (CUDA 12.2):
    pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

  NOTE: If no wheel exists for Python {sys.version_info.major}.{sys.version_info.minor} yet, try a specific version:
    pip install llama-cpp-python==0.3.2 --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

  ── OPTION B: Build from source (Windows, needs compiler) ──
  1. Install VS Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe
     (tick "Desktop development with C++")
  2. Open "x64 Native Tools Command Prompt" and run:
     pip install llama-cpp-python --no-cache-dir

  ── OPTION C: Continue with Ollama fallback ──
  Aria will use Ollama for local inference until llama-cpp is installed.
  Download Ollama from: https://ollama.ai

  Full docs: https://github.com/abetlen/llama-cpp-python#installation
  ─────────────────────────────────────────────────────
""")


# ── Other dependencies ─────────────────────────────────────────
def install_requirements():
    print("\n─── Installing core dependencies ───")

    # Install requirements one file at a time, skipping long-path errors
    for rf in ["requirements_core.txt", "requirements.txt"]:
        if not os.path.exists(rf):
            continue
        # Read and filter lines to install only what's needed
        with open(rf) as f:
            pkgs = [
                line.strip() for line in f
                if line.strip() and not line.startswith("#")
                and "llama-cpp-python" not in line
                and "CMAKE_ARGS" not in line
            ]
        # Install in small batches to avoid long-path cascades
        batch = []
        for pkg in pkgs:
            batch.append(pkg)
            if len(batch) >= 5:
                run([PY, "-m", "pip", "install", "--upgrade", "--quiet"] + batch,
                    check=False, quiet=True)
                batch = []
        if batch:
            run([PY, "-m", "pip", "install", "--upgrade", "--quiet"] + batch,
                check=False, quiet=True)
        print(f"  ✓ {rf} processed")

    # Essential extras
    for pkg in ["huggingface-hub", "psutil", "py-cpuinfo"]:
        run([PY, "-m", "pip", "install", pkg, "--quiet"], check=False, quiet=True)
    print("  ✓ Core packages ready")


# ── Verification ──────────────────────────────────────────────
def verify():
    print("\n─── Verifying installation ───")
    checks = [
        ("llama_cpp",       "llama-cpp-python"),
        ("huggingface_hub", "huggingface-hub"),
        ("fastapi",         "fastapi"),
        ("psutil",          "psutil"),
    ]
    all_ok = True
    for mod, label in checks:
        try:
            __import__(mod)
            print(f"  ✓ {label}")
        except ImportError:
            print(f"  ✗ {label}  ← MISSING")
            if mod == "llama_cpp":
                print("    Run the manual install command shown above.")
            all_ok = False
    return all_ok


def create_dirs():
    from pathlib import Path
    for d in ["data/hf_models", "data/logs", "data/uploads",
              "data/generated", "data/documents"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"\n  ✓ Data directories ready in: {Path('data').resolve()}")


def print_next_steps(backend, llama_ok):
    print()
    print("╔══════════════════════════════════════════════════════╗")
    if llama_ok:
        print("║  ✅ Setup complete!                                   ║")
    else:
        print("║  ⚠  Setup partial — llama-cpp-python missing          ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  1. Start Aria:  python main.py                      ║")
    print("║  2. Open:        http://localhost:8000               ║")
    print("║  3. HF Manager:  http://localhost:8000/hf            ║")
    print("╠══════════════════════════════════════════════════════╣")
    if backend == "cuda":
        print("║  GPU: NVIDIA CUDA — fast GPU inference ⚡             ║")
    elif backend == "metal":
        print("║  GPU: Apple Metal — fast GPU inference ⚡             ║")
    else:
        print("║  Mode: CPU-only (no GPU detected)                    ║")
    if not llama_ok:
        print("╠══════════════════════════════════════════════════════╣")
        print("║  ⚠  Install llama-cpp-python manually (see above)    ║")
        print("║     Aria will use Ollama as fallback until then      ║")
    print("╚══════════════════════════════════════════════════════╝")


def main():
    print(BANNER)

    # Upgrade pip silently
    run([PY, "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        check=False, quiet=True)

    # Windows: fix long path issues first
    if IS_WIN:
        fix_windows_long_paths()

    # Detect GPU
    backend, wheel_index, cmake_args = detect_gpu()

    # Install llama-cpp-python
    llama_ok = install_llama_cpp(backend, wheel_index, cmake_args)

    # Install all other packages
    install_requirements()

    # Create directories
    create_dirs()

    # Verify
    all_ok = verify()

    # Summary
    print_next_steps(backend, llama_ok)

    if not llama_ok:
        # Don't exit with error — Aria still runs with Ollama fallback
        sys.exit(0)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
