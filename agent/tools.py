# ============================================================
#  Agent Tools / Utilities
# ============================================================

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from agent.file_processor import compare_excel_text, excel_to_text, generate_excel, generate_ppt, generate_text, generate_word, read_file_for_llm

try:
    from langchain.tools import tool
except Exception:
    def tool(fn=None, **_kwargs):
        return (lambda f: f) if fn is None else fn


def search_web_text(query: str) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if results:
            return "\n\n".join(f"{r.get('title','')}\n{r.get('body','')}\n{r.get('href','')}" for r in results)
    except Exception:
        pass
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if results:
            return "\n\n".join(f"{r.get('title','')}\n{r.get('body','')}\n{r.get('href','')}" for r in results)
    except Exception:
        pass
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)
    except Exception:
        return "Search unavailable. Install or re-install ddgs."


@tool
def web_search(query: str) -> str:
    """Search the web for information about a query."""
    return f"Search results for '{query}':\n{search_web_text(query)}"


def resolve_file_path(file_path: str) -> Path:
    p = Path(file_path)
    if p.exists():
        return p
    for base in [cfg.UPLOADS_DIR, cfg.DOCUMENTS_DIR]:
        q = Path(base) / file_path
        if q.exists():
            return q
    return p


@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a file."""
    p = resolve_file_path(file_path)
    return read_file_for_llm(str(p)) if p.exists() else f"File not found: {file_path}"


@tool
def write_text_file(file_path: str, content: str) -> str:
    """Write content to a text file at the specified path."""
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File saved: {p} ({len(content):,} chars)"
    except Exception as e:
        return f"Error: {e}"


@tool
def list_files(directory: str = "") -> str:
    """List all files in a directory."""
    base = Path(directory) if directory else Path(cfg.DOCUMENTS_DIR)
    if not base.exists():
        return f"Directory not found: {base}"
    files = sorted(f for f in base.rglob("*") if f.is_file() and not f.name.startswith("."))
    if not files:
        return f"No files in {base}"
    return "Files:\n" + "\n".join(f"  {f.relative_to(base)} ({f.stat().st_size:,}B)" for f in files[:80])


@tool
def analyze_excel(file_path: str) -> str:
    """Analyze and convert an Excel file to text format."""
    p = resolve_file_path(file_path)
    return excel_to_text(str(p)) if p.exists() else f"File not found: {file_path}"


@tool
def compare_excel_files(file_path_a: str, file_path_b: str) -> str:
    """Compare two Excel files and return the differences."""
    a, b = resolve_file_path(file_path_a), resolve_file_path(file_path_b)
    if not a.exists(): return f"File A not found: {file_path_a}"
    if not b.exists(): return f"File B not found: {file_path_b}"
    return compare_excel_text(str(a), str(b))


@tool
def create_excel_report(json_data: str, output_filename: str) -> str:
    """Create an Excel report from JSON data."""
    try:
        return f"Excel file created: {generate_excel(json.loads(json_data), output_filename)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def create_word_document(title: str, sections_json: str, output_filename: str) -> str:
    """Create a Word document with the specified title and sections."""
    try:
        return f"Word document created: {generate_word(title, json.loads(sections_json), output_filename)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def create_presentation(title: str, slides_json: str, output_filename: str) -> str:
    """Create a presentation with the specified title and slides."""
    try:
        return f"Presentation created: {generate_ppt(title, json.loads(slides_json), output_filename)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def create_text_document(content: str, output_filename: str) -> str:
    """Create a text or markdown document with the specified content."""
    try:
        return f"Text file created: {generate_text(content, output_filename, markdown=output_filename.lower().endswith('.md'))}"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_python(code: str) -> str:
    """Execute Python code safely with restricted operations blocked."""
    blocked = ["os.system", "subprocess.run", "subprocess.Popen", "shutil.rmtree"]
    for token in blocked:
        if token in code:
            return f"Blocked: restricted operation '{token}'"
    try:
        out = io.StringIO(); local = {}
        with contextlib.redirect_stdout(out):
            exec(code, {"__builtins__": __builtins__, "json": json, "Path": Path, "datetime": datetime}, local)
        result = out.getvalue().strip()
        if not result and local:
            result = str(list(local.values())[-1])
        return result or "Executed successfully (no output)."
    except Exception as e:
        return f"Error: {e}"


@tool
def get_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("Date: %A, %B %d, %Y\nTime: %I:%M %p")


@tool
def save_note(title: str, content: str) -> str:
    """Save a note with the specified title and content."""
    notes = Path(cfg.LOG_DIR) / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(title or 'note').stem[:40]}.md"
    (notes / name).write_text(f"# {title}\n*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n{content}", encoding="utf-8")
    return f"Note saved: {name}"


@tool
def list_generated_files() -> str:
    """List all files that have been generated by the agent."""
    gen = Path(cfg.GENERATED_DIR)
    if not gen.exists() or not any(gen.iterdir()):
        return "No generated files yet."
    files = sorted(gen.rglob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
    return "Generated files:\n" + "\n".join(f"  {f.name} ({f.stat().st_size:,}B) — {datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}" for f in files[:50] if f.is_file())


ALL_TOOLS = [web_search, read_file, write_text_file, list_files, analyze_excel, compare_excel_files, create_excel_report, create_word_document, create_presentation, create_text_document, run_python, get_datetime, save_note, list_generated_files]
