# ============================================================
#  Document Ingestion Pipeline
# ============================================================

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from agent.file_processor import read_file_for_llm


def chunk_text(text: str) -> list[str]:
    """Chunks a large text into smaller parts to improve processing efficiency."""
    words  = text.split()
    chunks = []
    step   = cfg.CHUNK_SIZE - cfg.CHUNK_OVERLAP
    for i in range(0, max(len(words), 1), max(step, 1)):
        chunk = " ".join(words[i:i + cfg.CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks or [text[:500]]


def ingest_file(path: str, vector_memory) -> dict:
    """Ingests a file (e.g., text, PDF) into memory, converting its contents for later use."""
    p    = Path(path)
    text = read_file_for_llm(str(p))
    if not text or text.startswith("Error"):
        return {"success": False, "source": p.name, "error": text}
    chunks = chunk_text(text)
    for i, c in enumerate(chunks):
        vector_memory.save_document(c, p.name, i)
    print(f"  ✓ {p.name}: {len(chunks)} chunks")
    return {"success": True, "source": p.name, "chunks": len(chunks)}


def ingest_directory(vector_memory=None, directory: str = None) -> dict:
    """Ingests all files in a given directory into memory."""
    from agent.memory import VectorMemory
    if vector_memory is None:
        vector_memory = VectorMemory()
    directory = directory or cfg.DOCUMENTS_DIR
    path      = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    results = {"files_processed": 0, "total_chunks": 0, "errors": []}
    for fp in sorted(path.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in cfg.SUPPORTED_FORMATS:
            r = ingest_file(str(fp), vector_memory)
            if r["success"]:
                results["files_processed"] += 1
                results["total_chunks"]    += r.get("chunks", 0)
            else:
                results["errors"].append(r.get("error", ""))
    return results
