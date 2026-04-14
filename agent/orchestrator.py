# ============================================================
#  Orchestrator v6 — cloud-first hybrid, thinking steps
# ============================================================

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config as cfg
from agent.file_processor import (
    compare_excel_text, file_metadata, generate_excel,
    generate_ppt, generate_text, generate_word, read_file_for_llm
)
from agent.memory import AgentMemory
from agent.model_router import ChatResult, ModelRouter
from agent.tools import search_web_text

_WEB_KW     = ["latest","current","news","research","search","look up","web","online","today","breaking"]
_ANALYZE_KW = ["analyze","analyse","summarize","summary","compare","review","explain","describe","what is in"]


def _persona() -> str:
    return cfg.AGENT_PERSONA.replace("{name}", cfg.AGENT_NAME)


class PersonalAgent:
    def __init__(self):
        print(f"[{cfg.AGENT_NAME}] Initializing v6...")
        self.memory = AgentMemory()
        self.router = ModelRouter()
        print(f"[{cfg.AGENT_NAME}] Ready. Status: {self.router.status()}")

    # ─── internals ───────────────────────────────────────────

    def _system_prompt(self, user_msg: str) -> str:
        import datetime
        ctx = self.memory.recall(user_msg)
        prompt = _persona()
        prompt += f"\nDevice: {cfg.DEVICE_NAME}"
        prompt += f"\nTime: {datetime.datetime.now().strftime('%A %B %d %Y, %I:%M %p')}"
        if ctx:
            prompt += f"\n\n{ctx}"
        return prompt

    def _build_messages(self, user_prompt: str, system_prompt: str) -> list[dict]:
        return [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]

    def _extract_json(self, text: str) -> Optional[Any]:
        text = (text or "").strip()
        if not text:
            return None
        candidates = [text]
        candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S|re.I))
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            candidates.append(m.group(1))
        for cand in candidates:
            try:
                return json.loads(cand.strip())
            except Exception:
                pass
        return None

    def _detect_generation_target(self, message: str) -> Optional[str]:
        msg = (message or "").lower()
        if any(k in msg for k in ["powerpoint","presentation",".pptx","slides","slide deck","ppt"]): return "pptx"
        if any(k in msg for k in ["excel","spreadsheet",".xlsx","workbook","sheet"]): return "xlsx"
        if any(k in msg for k in ["word document",".docx","docx","formal report","proposal","letter","word file","summarize it into a word","summary into word"]): return "docx"
        if any(k in msg for k in ["markdown",".md"]): return "md"
        if any(k in msg for k in ["text file",".txt","plain text"]): return "txt"
        return None

    def _wants_web_search(self, message: str, attached_files: list[str]) -> bool:
        return not attached_files and any(k in (message or "").lower() for k in _WEB_KW)

    def _wants_file_analysis(self, message: str, attached_files: list[str]) -> bool:
        msg = (message or "").lower()
        return bool(attached_files) and (any(k in msg for k in _ANALYZE_KW) or self._detect_generation_target(message) is None)

    def _attached_context(self, attached_files: list[str], for_compare: bool = False) -> str:
        if not attached_files:
            return ""
        if for_compare and len(attached_files) >= 2:
            first_two = attached_files[:2]
            if all(Path(p).suffix.lower() in {".xlsx",".xls",".csv"} for p in first_two):
                return compare_excel_text(first_two[0], first_two[1])
        parts = []
        for fp in attached_files:
            text = read_file_for_llm(fp)
            parts.append(f"=== File: {Path(fp).name} ===\n{text[:cfg.FILE_PREVIEW_CHARS]}")
        return "\n\n".join(parts)

    def _record_artifact(self, path: str, source_prompt: str, model_used: str) -> dict:
        meta = file_metadata(path)
        self.memory.structured.add_artifact(
            meta["name"], meta["path"], meta["filetype"], meta["size"],
            source_prompt=source_prompt, model_used=model_used
        )
        return {**meta, "download_url": f"/api/download/{meta['name']}"}

    # ─── CLOUD-FIRST routing ──────────────────────────────────

    def _call_route(
        self, route: dict, messages: list[dict], *,
        prefer_json: bool = False, max_tokens: int = 1800,
        thinking_steps: Optional[list] = None
    ) -> tuple[str, list[str], float]:
        used_models, total_latency = [], 0.0
        ts = thinking_steps if thinking_steps is not None else []

        def _local(msgs):
            r = self.router.chat_local(msgs, route.get("local_model"), fmt="json" if prefer_json else None)
            used_models.append(f"{r.model} (local)")
            return r

        def _cloud(msgs):
            entry = route.get("cloud_entry") or {}
            r = self.router.chat_cloud(
                msgs, route.get("cloud_model"),
                base_url=entry.get("base_url") or self.router.openrouter_base(),
                api_key=entry.get("api_key") or self.router.openrouter_key(),
                max_tokens=max_tokens
            )
            used_models.append(f"{r.model} (cloud)")
            return r

        mode = route.get("mode")

        # Pure local
        if mode == "local":
            ts.append({"step": "🔧 Running local model", "detail": route.get("local_model","")})
            r = _local(messages)
            return r.content, used_models, r.latency_ms

        # Pure cloud
        if mode == "cloud":
            ts.append({"step": "☁️ Sending to cloud model", "detail": route.get("cloud_model","")})
            r = _cloud(messages)
            return r.content, used_models, r.latency_ms

        # Hybrid — CLOUD FIRST for speed
        if cfg.HYBRID_CLOUD_FIRST and route.get("cloud_model"):
            cloud_entry_data = route.get("cloud_entry") or {}
            has_key = bool(cloud_entry_data.get("api_key") or self.router.openrouter_key())
            if has_key:
                ts.append({"step": "☁️ Cloud-first — querying cloud model", "detail": route.get("cloud_model","")})
                try:
                    r = _cloud(messages)
                    total_latency += r.latency_ms
                    if r.content and len(r.content.strip()) > 5:
                        return r.content, used_models, total_latency
                    else:
                        ts.append({"step": "⚠️ Cloud returned empty — trying local", "detail": ""})
                except Exception as cloud_err:
                    ts.append({"step": "⚠️ Cloud failed — falling back to local", "detail": str(cloud_err)[:120]})
            else:
                ts.append({"step": "⚠️ No cloud API key — switching to local only", "detail": "Add your OpenRouter key in Settings → Providers"})

        # Local fallback (or local-only mode)
        if route.get("local_model"):
            ts.append({"step": "🔧 Running local model", "detail": route.get("local_model","")})
            try:
                local_result = _local(messages)
                total_latency += local_result.latency_ms
                if local_result.content and local_result.content.strip():
                    return local_result.content, used_models, total_latency
                ts.append({"step": "⚠️ Local model returned empty response", "detail": ""})
            except Exception as local_err:
                ts.append({"step": "❌ Local model failed", "detail": str(local_err)[:120]})

        raise RuntimeError(
            "No model responded. "
            "If using Ollama: make sure it is running (`ollama serve`) and a model is pulled (`ollama pull qwen2.5`). "
            "If using cloud: add your OpenRouter API key in Settings → Providers."
        )

    # ─── handlers ────────────────────────────────────────────

    def _generate_structured_output(
        self, message: str, route: dict, target: str,
        attachment_context: str, thinking_steps: list
    ) -> tuple[dict, list[str], float]:
        thinking_steps.append({"step": f"📐 Planning {target.upper()} structure", "detail": "Generating JSON schema for file content"})
        system_prompt = self._system_prompt(message)
        schema_hint = {
            "xlsx": '{"filename":"report.xlsx","sheet_name":"Data","rows":[{"Column1":"...","Column2":"..."}]}',
            "docx": '{"filename":"report.docx","title":"Report title","sections":[{"heading":"Executive Summary","content":"..."}]}',
            "pptx": '{"filename":"presentation.pptx","title":"Title","slides":[{"title":"Slide 1","bullets":["point 1"],"notes":"notes"}]}',
            "txt":  '{"filename":"notes.txt","content":"..."}',
            "md":   '{"filename":"notes.md","content":"# Heading\\n..."}',
        }[target]
        prompt = (
            f"User request:\n{message}\n\n"
            f"Supporting context:\n{attachment_context or '(none)'}\n\n"
            f"Return ONLY valid JSON for a {target} file.\nSchema shape:\n{schema_hint}\n"
            "Do not include markdown fences or explanation."
        )
        thinking_steps.append({"step": "🤖 Generating file content with AI", "detail": f"Model: {route.get('cloud_model') or route.get('local_model','')}"})
        text, used_models, latency = self._call_route(
            route, self._build_messages(prompt, system_prompt),
            prefer_json=True, thinking_steps=thinking_steps
        )
        data = self._extract_json(text)
        if data is None:
            retry = f"Convert the following into valid JSON only. No commentary.\n\nOriginal response:\n{text}\n\nSchema:\n{schema_hint}"
            text2, used2, latency2 = self._call_route(
                route, self._build_messages(retry, system_prompt),
                prefer_json=True, thinking_steps=thinking_steps
            )
            used_models.extend(used2); latency += latency2; data = self._extract_json(text2)
        if data is None or not isinstance(data, dict):
            raise RuntimeError("Could not get valid structured output from the model.")
        return data, used_models, latency

    def _handle_general_chat(self, message: str, route: dict, thinking_steps: list) -> dict:
        thinking_steps.append({"step": "💬 Preparing response", "detail": "General conversation"})
        text, used_models, latency = self._call_route(
            route, self._build_messages(message, self._system_prompt(message)),
            thinking_steps=thinking_steps
        )
        return {"response": text, "tools_used": [], "generated_files": [],
                "model_used": " + ".join(used_models) if used_models else "unknown", "latency_ms": latency}

    def _handle_web_research(self, message: str, route: dict, thinking_steps: list) -> dict:
        thinking_steps.append({"step": "🔍 Searching the web", "detail": f"Query: {message[:80]}"})
        results = search_web_text(message)
        thinking_steps.append({"step": "📰 Processing search results", "detail": f"Found {len(results.splitlines())} result lines"})
        prompt = (
            f"User request: {message}\n\nWeb search results:\n{results}\n\n"
            "Present the findings in a rich, well-structured Markdown format. "
            "Use a table when listing multiple sources or items. "
            "Use **bold** for key facts. Add a brief summary at the top. "
            "Do not fabricate facts not present in the search results."
        )
        text, used_models, latency = self._call_route(
            route, self._build_messages(prompt, self._system_prompt(message)),
            thinking_steps=thinking_steps
        )
        return {"response": text, "tools_used": ["web_search"], "generated_files": [],
                "model_used": " + ".join(used_models) if used_models else "unknown", "latency_ms": latency}

    def _handle_file_analysis(self, message: str, attached_files: list[str], route: dict, thinking_steps: list) -> dict:
        thinking_steps.append({"step": "📂 Reading attached files", "detail": ", ".join(Path(f).name for f in attached_files)})
        compare_mode = "compare" in message.lower() and len(attached_files) >= 2
        file_context = self._attached_context(attached_files, for_compare=compare_mode)
        thinking_steps.append({"step": "🧠 Analysing file content", "detail": f"Context size: {len(file_context)} chars"})
        prompt = (
            f"User instruction:\n{message}\n\n"
            f"Attached file context:\n{file_context}\n\n"
            "Provide a thorough, well-structured Markdown analysis. "
            "Use tables for lists of data, **bold** for key figures, and bullet points for summaries. "
            "If comparing, highlight the key differences first. "
            "Include statistics and counts where relevant."
        )
        text, used_models, latency = self._call_route(
            route, self._build_messages(prompt, self._system_prompt(message)),
            max_tokens=2400, thinking_steps=thinking_steps
        )
        return {"response": text, "tools_used": ["file_analysis"], "generated_files": [],
                "model_used": " + ".join(used_models) if used_models else "unknown", "latency_ms": latency}

    def _handle_file_generation(self, message: str, attached_files: list[str], route: dict, target: str, thinking_steps: list) -> dict:
        thinking_steps.append({"step": f"🛠️ Creating {target.upper()} file", "detail": "Preparing content structure"})
        payload, used_models, latency = self._generate_structured_output(
            message, route, target, self._attached_context(attached_files), thinking_steps
        )
        thinking_steps.append({"step": "💾 Writing file to disk", "detail": f"Type: {target.upper()}"})
        if target == "xlsx":
            path = generate_excel(
                payload.get("rows") or payload.get("data") or [],
                payload.get("filename") or "generated_report.xlsx",
                sheet_name=payload.get("sheet_name") or "Data"
            )
        elif target == "docx":
            path = generate_word(
                payload.get("title") or "Generated Report",
                payload.get("sections") or [],
                payload.get("filename") or "generated_report.docx"
            )
        elif target == "pptx":
            path = generate_ppt(
                payload.get("title") or "Presentation",
                payload.get("slides") or [],
                payload.get("filename") or "presentation.pptx"
            )
        elif target == "md":
            path = generate_text(payload.get("content") or "", payload.get("filename") or "notes.md", markdown=True)
        else:
            path = generate_text(payload.get("content") or "", payload.get("filename") or "notes.txt", markdown=False)
        artifact = self._record_artifact(path, message, " + ".join(used_models))
        lines = [
            f"✅ I created **{artifact['name']}** successfully.",
            "",
            "**The file is ready for download** — click the download button below or visit the Files section.",
        ]
        if target == "xlsx" and payload.get("rows"):
            lines.append(f"- **Rows:** {len(payload.get('rows') or [])}")
        if target == "pptx" and payload.get("slides"):
            lines.append(f"- **Slides:** {len(payload.get('slides') or [])}")
        if target == "docx" and payload.get("sections"):
            lines.append(f"- **Sections:** {len(payload.get('sections') or [])}")
        return {
            "response": "\n".join(lines),
            "tools_used": [f"create_{target}"],
            "generated_files": [artifact],
            "model_used": " + ".join(used_models) if used_models else "unknown",
            "latency_ms": latency
        }

    # ─── public chat ─────────────────────────────────────────

    def chat(
        self, user_message: str, attached_files: Optional[list[str]] = None,
        verbose: bool = False, execution_mode: Optional[str] = None,
        local_model: Optional[str] = None, cloud_model: Optional[str] = None
    ) -> dict:
        attached_files = attached_files or []
        started        = time.perf_counter()
        thinking_steps: list[dict] = []
        route = self.router.choose_route(
            user_message, attached_files,
            execution_mode=execution_mode,
            local_model=local_model,
            cloud_model=cloud_model
        )
        thinking_steps.append({"step": "🔀 Route selected", "detail": f"Mode: {route.get('mode')} | Task: {route.get('task_type','general')}"})
        try:
            target = self._detect_generation_target(user_message)
            if target:
                result = self._handle_file_generation(user_message, attached_files, route, target, thinking_steps)
            elif self._wants_web_search(user_message, attached_files):
                result = self._handle_web_research(user_message, route, thinking_steps)
            elif self._wants_file_analysis(user_message, attached_files):
                result = self._handle_file_analysis(user_message, attached_files, route, thinking_steps)
            else:
                result = self._handle_general_chat(user_message, route, thinking_steps)
        except Exception as e:
            error_msg = str(e)[:350]
            suggestions = []
            if "No cloud API key" in error_msg or "API key" in error_msg:
                suggestions.append("Cloud API is not configured.\n1. Get a FREE OpenRouter API key at https://openrouter.ai/keys\n2. Add it in Settings → Provider Settings → OpenRouter API Key")
            elif "Ollama" in error_msg or "local" in error_msg.lower():
                suggestions.append("Make sure Ollama is running. Use the **Start Ollama** button in Settings, or run `ollama serve` in a terminal.")
            else:
                suggestions.append("Check your model configuration in Settings.")
            result = {
                "response": f"⚠️ **Error:** {error_msg}\n\n{'  '.join(suggestions)}",
                "tools_used": [], "generated_files": [],
                "model_used": "error",
                "latency_ms": round((time.perf_counter()-started)*1000, 2)
            }
            self.memory.structured.log("ERROR","Chat failed",{"error":str(e)[:500]})

        result["execution_mode"]  = route.get("mode")
        result["thinking_steps"]  = thinking_steps
        result["latency_ms"]      = round(result.get("latency_ms") or (time.perf_counter()-started)*1000, 2)
        self.memory.remember(user_message, result["response"])
        self.memory.structured.log("INFO","Chat",{
            "mode":    route.get("mode"),
            "model":   result.get("model_used"),
            "tools":   result.get("tools_used"),
            "latency_ms": result.get("latency_ms"),
            "generated": [f.get("name") for f in result.get("generated_files",[])]
        })
        return result

    # ─── tasks ───────────────────────────────────────────────

    def add_task(self, title, description="", priority="normal", scheduled_at=None, reminder_minutes=15):
        return self.memory.structured.add_task(title, description, priority, scheduled_at, reminder_minutes)

    def get_tasks(self, status=None):
        return self.memory.structured.get_tasks(status)

    def update_task(self, tid, status=None, result=None, **extra):
        self.memory.structured.update_task(tid, status=status, result=result, **extra)

    def delete_task(self, tid):
        self.memory.structured.delete_task(tid)

    def get_task_reminders(self):
        due = self.memory.structured.get_due_task_reminders()
        for task in due:
            self.memory.structured.mark_task_reminded(task["id"])
        return due

    def run_task(self, tid, execution_mode=None, local_model=None, cloud_model=None):
        task = self.memory.structured.get_task(tid)
        if not task:
            raise RuntimeError("Task not found")
        prompt = task["title"] + (f"\n\nTask details:\n{task['description']}" if task.get("description") else "")
        self.memory.structured.update_task(tid, status="running")
        result = self.chat(prompt, execution_mode=execution_mode, local_model=local_model, cloud_model=cloud_model)
        final_status = "done" if "error" not in (result.get("model_used") or "") else "failed"
        self.memory.structured.mark_task_run(tid, result=result.get("response"), status=final_status)
        self.memory.structured.log("INFO","Task run",{"task_id":tid,"status":final_status,"model":result.get("model_used")})
        return result

    def learn_fact(self, key, value):
        self.memory.structured.set_fact(key, value)

    def set_preference(self, key, value):
        self.memory.structured.set_preference(key, value)
        self.router.refresh()

    def stats(self):
        mem    = self.memory.stats()
        counts = {}
        for t in self.memory.structured.get_tasks():
            counts[t["status"]] = counts.get(t["status"],0) + 1
        return {
            "memory":   mem,
            "tasks":    counts,
            "models":   self.router.status(),
            "artifacts": len(self.memory.structured.get_artifacts(1000))
        }
