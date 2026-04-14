# Aria Dashboard — Fixes Applied (v2.2)

## 1. ✅ Startup Aborted Popup — FIXED
**Problem:** `init()` called `refreshAll()` immediately, hitting agent APIs before the agent
was ready, causing a toast popup "Startup error: Agent is still starting."

**Fix:** Added `waitForAgent()` — polls `/api/ping` up to 40× with 1s interval. Shows an
animated loading overlay during startup. If the agent isn't ready in time, a friendly
in-chat message is shown instead of an error popup. No more "Startup aborted" toast.

## 2. 🎨 Dashboard Redesign — DONE
**Changes:**
- Full pastel green Claude-inspired theme: soft greens (#edfbf3, #d1fae5, #a7f3d0)
- Gradient brand header with green-to-blue title text
- Compact sidebar (230px) with coloured active state
- All pages fit in a single viewport — no body scroll
- Chat, content panels, and lists scroll *internally* within their containers
- Color-coded latency badges: green < 600ms, amber < 1800ms, red ≥ 1800ms

## 3. 🔌 Connection Status — IMPROVED
**Problem:** Local showing 2088ms and cloud 896ms with no visual distinction; hybrid
connection status was not clearly displayed.

**Fix:**
- Added persistent **Connection Bar** below topbar showing all 4 providers at a glance
- Latency badges are colour-coded (green/amber/red) so high latency is obvious vs error
- Added dedicated **Hybrid status** pill showing whether both, one, or neither provider is active
- Sidebar latency badges show real-time ms for local + cloud
- Settings page now includes a "Hybrid mode" mini-card explaining current routing state

## 4. ⚡ Hybrid Mode — FIXED DISPLAY & LOGIC
**Problem:** Hybrid (local+cloud) routing wasn't clearly shown; mode dropdown didn't
update sidebar or show hybrid-specific UI.

**Fix:**
- Hybrid route strip appears in chat toolbar when "Hybrid" mode is selected
- Sidebar "Active Route" card updates on every save showing local+cloud models in use
- All API calls use `Promise.allSettled` so a single provider failure doesn't break init
- Connection bar shows "Both online", "Local only", "Cloud only" or "Both offline"
