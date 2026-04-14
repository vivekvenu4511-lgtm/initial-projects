# Aria Personal Agent v6 — Upgrade Notes

## What's New in v6

### 1. Rich Interactive Responses (Claude-style)
- All responses now render **full Markdown**: headings, tables, bold, bullet lists, code blocks with syntax highlighting
- News/web search results are formatted as structured tables with source links
- File analysis includes statistics, highlighted key figures, and visual summaries
- A "How I worked on this" collapsible **thinking steps panel** shows each reasoning step

### 2. 🧠 Intelligence Mode
- New **Intelligence** button in the chat toolbar
- Automatically analyses your prompt (coding / reasoning / analysis / general) and selects the **best available model** for that task type
- Shows an "Intelligence Active" badge on responses and the route taken
- Preview the selected route via the internal API: `POST /api/intelligence/route`

### 3. ⚡ Cloud-First Hybrid Routing (Faster Responses)
- In Hybrid mode, **cloud is tried first** (it's 3-5× faster: ~400ms vs 2000ms+ local)
- Local model is used as fallback only if cloud fails or returns an empty result
- Eliminates the previous double-round-trip (local → cloud refine) for most queries
- Response time should drop from ~50s to under 5s for typical cloud-capable prompts

### 4. 🎤 Voice Input & Output
- **Voice input**: Click the 🎤 microphone button to speak your message (Chrome/Edge)
- **Voice output**: Click 🔊 to toggle spoken responses after every AI reply
- Stop speaking at any time with the "■ Stop" button that appears
- Admin panel lets you choose male or female voice, plus a specific system voice

### 5. 📁 Generated Files — Always Downloadable
- Files (Word, Excel, PowerPoint) now show a prominent **📥 Download** button directly in the chat message
- Files are also listed in the **Files** page for later retrieval
- The file path issue from v5 is fixed — files are always written to `data/generated/`

### 6. 🤖 Models Page (New)
- Dedicated **Models** section in the sidebar
- Shows capability cards for every known local and cloud model
- Filter by: General, Coding, Reasoning, Analysis, Fast, Creative
- **Recommendations grid** suggests the best model for: summarising, code, reasoning, creative, fast tasks, multilingual
- Adding a custom model will also auto-populate capability info if it matches a known ID

### 7. 🦙 Ollama Auto-Start
- On startup, `main.py` checks if Ollama is running and **starts it automatically**
- A dedicated **Settings → Ollama** tab has a "▶ Start Ollama" button for manual starts
- Works on Windows, macOS (via `open -a Ollama`), and Linux
- Reduces connection errors and program hanging on startup

### 8. 🔒 Admin / Advanced Panel (Password Protected)
- Go to **Settings → Advanced** tab
- Default password: **`aria2025`**
- **Agent tab**: Rename the agent (changes sidebar and chat name)
- **Voice tab**: Choose male/female voice, pick a specific system voice, preview it
- **Security tab**: Change the admin password
- Password is hashed (SHA-256) and stored in the database

### 9. Settings Tabs
- Settings page is now tabbed: **Providers | Models | Ollama | Advanced**
- Cleaner and easier to navigate

---

## Upgrade Steps from v5

1. **Backup** your `data/` folder (contains your database and knowledge base)
2. Copy your `data/` folder into the new `07.Aria_Agent_v6/data/` directory
3. Install updated dependencies: `pip install -r requirements.txt --upgrade`
4. Run: `python main.py` (or use the run script for your OS)

### Re-using your OpenRouter API key
Your key is stored in the database — it will carry over automatically if you copied the `data/` folder.

---

## Configuration Reference

Key settings in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `AGENT_NAME` | `Aria` | Agent display name (overridden by admin panel) |
| `HYBRID_CLOUD_FIRST` | `True` | Cloud-first in hybrid mode for speed |
| `CLOUD_MODEL_BALANCED` | `meta-llama/llama-3.3-70b-instruct:free` | Default cloud model |
| `CLOUD_MODEL_CODING` | `qwen/qwen-2.5-coder-32b-instruct:free` | Code tasks |
| `CLOUD_MODEL_REASON` | `deepseek/deepseek-r1:free` | Deep reasoning |
| `CLOUD_MODEL_FAST` | `google/gemma-3-4b-it:free` | Fast simple queries |
| `ADMIN_PASSWORD_HASH` | SHA-256 of `aria2025` | Default admin password |

---

## API Endpoints Added in v6

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Now accepts `intelligence_mode: bool` |
| `POST` | `/api/intelligence/route` | Preview model selection for a prompt |
| `GET` | `/api/models/capabilities` | Full capability catalogue |
| `GET/POST` | `/api/ollama/status` | Ollama status |
| `POST` | `/api/ollama/start` | Auto-start Ollama |
| `POST` | `/api/admin/verify` | Verify admin password |
| `GET/POST` | `/api/admin/settings` | Read/write admin settings |

---

## Troubleshooting

**Response still slow?**
- Make sure Hybrid mode is selected (not Local-only)
- Set your OpenRouter API key in Settings → Providers

**Voice input not working?**
- Requires Chrome or Edge browser (Firefox doesn't support Web Speech API)
- HTTPS is required on remote deployments (works on localhost without HTTPS)

**Files not downloading?**
- Check the Files page — all generated files are listed there
- The download link is `/api/download/<filename>`

**Admin password forgotten?**
- Default password is always `aria2025` until changed
- To reset: delete the `admin_password_hash` row from `data/agent.db` preferences table
