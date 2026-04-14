# ============================================================
#  Scheduler — Background jobs
# ============================================================

import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from datetime import datetime
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

_agent = None


def set_agent(agent):
    """Sets the agent instance for scheduling tasks and background jobs."""
    global _agent
    _agent = agent


def job_morning_summary():
    """Generates and sends a summary of the previous day's activities each morning."""
    if not _agent: return
    tasks  = _agent.memory.structured.get_tasks("pending")
    tlist  = "\n".join(f"- {t['title']} ({t['priority']})" for t in tasks) or "No pending tasks."
    prompt = (f"Today is {datetime.now().strftime('%A, %B %d, %Y')}.\n"
              f"Pending tasks:\n{tlist}\n\n"
              "Give me a concise morning briefing: top priorities, any time-sensitive items, "
              "and a brief plan for today. Be practical, not verbose.")
    result = _agent.chat(prompt)
    out_dir = Path(cfg.LOG_DIR) / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    (out_dir / f"{date_str}.md").write_text(
        f"# Daily Briefing — {date_str}\n\n{result['response']}", encoding="utf-8"
    )
    print(f"[Scheduler] Morning summary saved.")


def job_reingest():
    """Re-ingests documents and data periodically to keep the knowledge base up to date."""
    from ingestion.ingest import ingest_directory
    if not _agent: return
    print("[Scheduler] Re-indexing documents...")
    ingest_directory(_agent.memory.vector)


def job_cleanup():
    """Performs periodic cleanup of temporary files or data to optimize system performance."""
    try:
        with sqlite3.connect(cfg.SQLITE_DB) as c:
            c.execute("DELETE FROM logs WHERE created_at < datetime('now', '-30 days')")
            c.execute("DELETE FROM activity WHERE recorded_at < datetime('now', '-30 days')")
    except Exception as e:
        print(f"[Scheduler] Cleanup error: {e}")


def create_scheduler() -> BackgroundScheduler:
    """Creates a scheduler instance to manage background jobs and tasks."""
    scheduler = BackgroundScheduler(timezone=cfg.TIMEZONE)
    scheduler.add_job(job_morning_summary, CronTrigger(hour=cfg.DAILY_SUMMARY_HOUR, minute=0),
                      id="morning_summary", replace_existing=True)
    scheduler.add_job(job_reingest, CronTrigger(hour=2, minute=0),
                      id="reingest", replace_existing=True)
    scheduler.add_job(job_cleanup, CronTrigger(day_of_week="sun", hour=3, minute=0),
                      id="cleanup", replace_existing=True)
    return scheduler
