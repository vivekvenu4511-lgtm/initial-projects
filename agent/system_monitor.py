# ============================================================
#  System Monitor — Tracks active apps, learns usage patterns
#  Provides context-aware suggestions
# ============================================================

import time, threading, platform, sys, os
from datetime import datetime
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg

PLATFORM = platform.system()


def get_active_window() -> tuple[str, str]:
    """Returns (app_name, window_title). Cross-platform."""
    try:
        if PLATFORM == "Darwin":
            import subprocess
            script = """
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
                set frontWindow to ""
                try
                    set frontWindow to name of first window of (first application process whose frontmost is true)
                end try
            end tell
            return frontApp & "|||" & frontWindow
            """
            result = subprocess.run(["osascript", "-e", script],
                                     capture_output=True, text=True, timeout=2)
            parts  = result.stdout.strip().split("|||")
            return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")

        elif PLATFORM == "Windows":
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            buf  = ctypes.create_unicode_buffer(512)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
            title = buf.value
            # Get process name
            pid  = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                import psutil
                proc = psutil.Process(pid.value)
                return proc.name().replace(".exe", ""), title
            except Exception:
                return "Unknown", title

        else:
            # Linux / fallback — try xdotool
            import subprocess
            try:
                wid  = subprocess.check_output(["xdotool", "getactivewindow"],
                                               timeout=2).decode().strip()
                name = subprocess.check_output(
                    ["xdotool", "getwindowname", wid], timeout=2
                ).decode().strip()
                pid  = subprocess.check_output(
                    ["xdotool", "getwindowpid", wid], timeout=2
                ).decode().strip()
                import psutil
                proc = psutil.Process(int(pid))
                return proc.name(), name
            except Exception:
                return "Unknown", ""
    except Exception:
        return "Unknown", ""


# ── Suggestion rules ─────────────────────────────────────────

APP_SUGGESTIONS = {
    "Microsoft Excel": [
        "I can analyze your spreadsheet — just upload it or ask me a question.",
        "Need formulas, pivot summaries, or data comparisons? I'm ready.",
        "I can generate a formatted Excel report from any data you give me.",
    ],
    "Excel": [
        "I can analyze your spreadsheet — just upload it or ask me a question.",
    ],
    "Microsoft Word": [
        "I can improve, format, or summarize your document.",
        "Need a professional report or template? Just ask.",
    ],
    "Pages": [
        "I can help you write or improve your document.",
    ],
    "Microsoft PowerPoint": [
        "I can generate a full presentation for you — just give me a topic or outline.",
    ],
    "Keynote": [
        "I can help you build a presentation. Give me a topic and I'll create slides.",
    ],
    "Google Chrome": [
        "I can research topics, summarize articles, or help you draft web content.",
    ],
    "Safari": [
        "Need help researching? I can search the web for you.",
    ],
    "Terminal": [
        "I can write shell scripts, debug commands, or explain terminal output.",
    ],
    "Code": [
        "I can review your code, suggest improvements, or generate new functions.",
    ],
    "PyCharm": [
        "I can help debug, refactor, or document your Python code.",
    ],
    "Xcode": [
        "I can help with Swift code, debugging, or architecture questions.",
    ],
    "Outlook": [
        "I can draft professional emails, summarize threads, or schedule tasks.",
    ],
    "Mail": [
        "I can draft or improve emails for you.",
    ],
}


class SystemMonitor:
    def __init__(self, memory=None):
        self.memory          = memory
        self._running        = False
        self._thread         = None
        self._current_app    = ""
        self._current_window = ""
        self._app_start      = time.time()
        self._suggestions    = []
        self._last_suggest   = 0
        self._lock           = threading.Lock()

    def start(self):
        if not cfg.MONITOR_ENABLED:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                app, window = get_active_window()
                now = time.time()

                with self._lock:
                    if app != self._current_app:
                        # Log previous app's duration
                        duration = int(now - self._app_start)
                        if self._current_app and duration > 5 and self.memory:
                            self.memory.structured.log_activity(
                                self._current_app, self._current_window, duration
                            )
                        self._current_app    = app
                        self._current_window = window
                        self._app_start      = now

                    # Suggestion check
                    if now - self._last_suggest > cfg.MONITOR_SUGGEST_SECS:
                        self._generate_suggestion(app)
                        self._last_suggest = now

            except Exception:
                pass
            time.sleep(cfg.MONITOR_POLL_SECS)

    def _generate_suggestion(self, app: str):
        for key, suggestions in APP_SUGGESTIONS.items():
            if key.lower() in app.lower():
                import random
                suggestion = random.choice(suggestions)
                with self._lock:
                    self._suggestions.append({
                        "app": app,
                        "text": suggestion,
                        "time": datetime.now().isoformat(),
                    })
                # Keep only last 5
                self._suggestions = self._suggestions[-5:]
                break

    def get_current_app(self) -> str:
        with self._lock:
            return self._current_app

    def get_suggestions(self) -> list[dict]:
        with self._lock:
            result = list(self._suggestions)
            self._suggestions = []   # Clear after reading
        return result

    def get_status(self) -> dict:
        with self._lock:
            return {
                "active_app": self._current_app,
                "active_window": self._current_window[:60],
                "monitoring": self._running,
            }

    def get_usage_report(self) -> list[dict]:
        if not self.memory:
            return []
        return self.memory.structured.get_activity_summary(7)
