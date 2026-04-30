"""Streamlit UI for the Rideshare Ops RL Gym.

5 pages:
  * Run Episode  — pick task / agent / model / perturbations, run with live
                   tool-call streaming, see verifier breakdown
  * Browse Runs  — list past runs, view scorecards + trajectories
  * Task Catalog — what each of the 12 tasks does
  * Live World   — peek at a tenant's full state (drivers, riders, trips, KPIs)
  * Map View     — matplotlib map of the city showing driver positions
                   (status-coloured), pending requests, surge zones; the
                   "killer demo" view for real-time tasks

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import pandas as pd
import streamlit as st

from rideshare_gym.core.adversarial import FixtureMutator, Perturbation
from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.core.types import ToolCall
from rideshare_gym.rideshare_sandbox import in_process_sandbox_factory
from rideshare_gym.tasks import REGISTRY as TASK_REGISTRY
from rideshare_gym.tasks import all_task_ids

PERT_PRESETS = {
    "latency": {"endpoint": "*", "p": 0.3, "ms": 200},
    "rate_limit": {"p": 0.05, "status": 429, "retry_after": 1},
    "partial_failure": {"action": "*", "step": "*", "p": 0.10},
    "system_outage_partial": {"router": "/api/v1/safety/", "p": 1.0},
}
RUNS_DIR = _PROJ_ROOT / "runs"

st.set_page_config(
    page_title="Rideshare Ops RL Gym",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _shared_factory():
    return in_process_sandbox_factory(tenant_prefix="ui")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("Rideshare Ops RL Gym")
st.sidebar.caption("Real-time ride-sharing simulator + 12 ops tasks + "
                    "ride-sharing-specific adversarial mode")

PAGES = ["Run Episode", "Browse Runs", "Task Catalog", "Live World", "Map View"]
page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### Status")
las_set = bool(os.environ.get("LAS_API_TOKEN") or os.environ.get("OPENAI_API_KEY"))
anth_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
st.sidebar.markdown(
    f"- LAS / OpenAI key: {'set' if las_set else 'NOT set'}  \n"
    f"- Anthropic key: {'set' if anth_set else 'NOT set'}"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fmt_args(args: dict, max_chars: int = 220) -> str:
    s = json.dumps(args, default=str)
    return s if len(s) <= max_chars else s[:max_chars] + "..."


def _build_perturbations(kinds: list[str]) -> list[Perturbation]:
    return [Perturbation(kind=k, params=PERT_PRESETS[k]) for k in kinds]


def _build_agent(agent_choice: str, model: str, base_url: str | None, verbose: bool):
    if agent_choice == "gold_oracle":
        from agents.gold_oracle import GoldOracleAgent
        return GoldOracleAgent()
    if agent_choice == "litellm":
        from agents.litellm_agent import LiteLLMAgent
        return LiteLLMAgent(model=model, base_url=base_url or
                            "https://llm-west.ncsu-las.net/v1", verbose=verbose)
    if agent_choice == "claude_baseline":
        from agents.claude_baseline import ClaudeBaselineAgent
        return ClaudeBaselineAgent(model=model, verbose=verbose)
    raise ValueError(f"unknown agent: {agent_choice}")


# --------------------------------------------------------------------------- #
# Map rendering
# --------------------------------------------------------------------------- #

def render_map(world, *, title: str = "City") -> bytes:
    """Render a matplotlib map of the world. Returns PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=96)
    ax.set_xlim(world.city.bounds[0] - 1, world.city.bounds[2] + 1)
    ax.set_ylim(world.city.bounds[1] - 1, world.city.bounds[3] + 1)
    ax.set_aspect("equal")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    # Zones (filled circles, surge-coloured).
    for z in world.city.zones:
        surge = float(world.surge_zones.get(z.id, 1.0))
        # Surge → colour: 1.0 = gray, 1.5 = yellow, 2+ = orange/red
        if surge >= 2.0:
            color = "#ff4040"
        elif surge >= 1.5:
            color = "#ffaa00"
        elif surge > 1.0:
            color = "#ffd24d"
        else:
            color = "#444"
        c = plt.Circle(z.centroid, z.radius_km, color=color, alpha=0.18, zorder=1)
        ax.add_patch(c)
        ax.text(z.centroid[0], z.centroid[1] - z.radius_km - 0.4,
                f"{z.name or z.id}\nx{surge:.2f}",
                ha="center", color="white", fontsize=8)

    # Drivers, coloured by status.
    status_colors = {
        "idle": "#3ddc84",
        "dispatched": "#ffaa00",
        "picking_up": "#00b3ff",
        "in_trip": "#ff4d6d",
        "offline": "#666",
        "break": "#888",
    }
    for d in world.drivers.values():
        c = status_colors.get(d.status.value, "#fff")
        ax.scatter([d.location[0]], [d.location[1]], c=c, s=20, zorder=3,
                   edgecolors="white", linewidths=0.3)

    # Pending request pickups as cyan diamonds.
    from rideshare_gym.world.trips import TripStatus
    pending = [t for t in world.trips.values() if t.status == TripStatus.REQUESTED]
    for t in pending[:200]:
        ax.scatter([t.pickup[0]], [t.pickup[1]],
                    c="#00ffff", s=12, marker="D", zorder=4,
                    edgecolors="white", linewidths=0.2)

    ax.set_title(f"{title} — t={world.clock.now:.0f}s "
                 f"({len(world.drivers)} drivers, "
                 f"{sum(1 for t in world.trips.values() if t.status.value == 'requested')} pending)",
                 color="white")

    legend = [
        mpatches.Patch(color=status_colors["idle"], label="idle driver"),
        mpatches.Patch(color=status_colors["dispatched"], label="dispatched"),
        mpatches.Patch(color=status_colors["picking_up"], label="picking up"),
        mpatches.Patch(color=status_colors["in_trip"], label="in trip"),
        mpatches.Patch(color="#00ffff", label="pending request"),
        mpatches.Patch(color="#ff4040", label="surge >= 2x"),
    ]
    ax.legend(handles=legend, loc="upper left", facecolor="#0e1117",
              edgecolor="white", labelcolor="white", fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# PAGE: Run Episode
# --------------------------------------------------------------------------- #

def page_run_episode():
    st.title("Run an episode")
    st.caption("Pick a task, an agent, optional perturbations; watch the trajectory stream live.")

    cfg, live = st.columns([1, 2])

    with cfg:
        st.markdown("### Configuration")
        task_id = st.selectbox(
            "Task", all_task_ids(),
            format_func=lambda t: t.replace("rideshare/", ""),
        )
        agent_choice = st.radio("Agent", ["gold_oracle", "litellm", "claude_baseline"])
        model = ""
        base_url = None
        if agent_choice == "litellm":
            model = st.text_input("Model", value="openai/gpt-4o")
            base_url = st.text_input("Base URL",
                                       value="https://llm-west.ncsu-las.net/v1")
            if not las_set:
                st.error("Set `LAS_API_TOKEN` (or `OPENAI_API_KEY`) before running.")
        elif agent_choice == "claude_baseline":
            model = st.text_input("Model", value="claude-opus-4-7")
            if not anth_set:
                st.error("Set `ANTHROPIC_API_KEY` before running.")
        seed = st.number_input("Seed", value=0, step=1, min_value=0)
        adversarial = st.multiselect(
            "Adversarial perturbations",
            list(PERT_PRESETS.keys()),
            default=[],
        )
        run_btn = st.button("Start episode", type="primary", use_container_width=True)

    with live:
        st.markdown("### Trajectory (live)")
        if not run_btn:
            st.info(
                "Click **Start episode**. Trajectory streams here step by step. "
                "Easy tasks finish in <1s; H1 realtime_dispatch_window can run "
                "for 200+ tool calls / 60-90s wall clock."
            )
            return

        base_task = TASK_REGISTRY[task_id](seed=int(seed))
        perts = _build_perturbations(adversarial)
        task = FixtureMutator(base_task, perts) if perts else base_task
        sandbox_factory = _shared_factory()
        env = GymEnvironment(task=task, sandbox_factory=sandbox_factory)

        status_placeholder = st.empty()
        steps_placeholder = st.empty()
        events_placeholder = st.empty()

        steps: list[dict] = []
        events: list[dict] = []

        def render_steps():
            if not steps:
                return
            df = pd.DataFrame(steps)
            steps_placeholder.dataframe(df, use_container_width=True, hide_index=True)

        def on_step(step_idx, tc, obs, reward, terminated, info):
            steps.append({
                "#": step_idx, "tool": tc.name,
                "args": _fmt_args(tc.arguments),
                "ok": "yes" if info["tool_ok"] else "no",
                "reward": round(reward, 3),
                "term": "yes" if terminated else "",
                "latency_ms": round(info.get("tool_latency_ms", 0.0), 1),
                "error": (info.get("tool_error") or "")[:80],
            })
            if step_idx % 5 == 0 or terminated:
                render_steps()

        def on_event(ev):
            events.append({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "event": ev.get("event", "?"),
                "detail": json.dumps(
                    {k: v for k, v in ev.items() if k != "event"},
                    default=str)[:160],
            })
            with events_placeholder.expander(f"Agent events ({len(events)})"):
                st.dataframe(pd.DataFrame(events), use_container_width=True,
                             hide_index=True)

        try:
            agent = _build_agent(agent_choice, model, base_url, verbose=False)
        except Exception as e:
            st.error(f"Could not initialise agent: {type(e).__name__}: {e}")
            return

        t0 = time.perf_counter()
        with status_placeholder:
            with st.status("Running episode...", expanded=True) as status:
                try:
                    traj = agent.run(env, on_step=on_step, on_event=on_event)
                    elapsed = time.perf_counter() - t0
                    if traj.success:
                        status.update(
                            label=f"SUCCESS — reward {traj.final_reward:.2f} in "
                                  f"{len(traj.steps)} steps, {elapsed:.1f}s",
                            state="complete",
                        )
                    else:
                        status.update(
                            label=f"FAILED — {traj.error_category} after "
                                  f"{len(traj.steps)} steps, "
                                  f"reward {traj.final_reward:.2f}, {elapsed:.1f}s",
                            state="error",
                        )
                except Exception as e:
                    status.update(label=f"CRASH: {type(e).__name__}: {e}",
                                  state="error")
                    st.exception(e)
                    return

        render_steps()

        # Verifier breakdown.
        st.markdown("### Verifier")
        last_step_info = traj.steps[-1].info if traj.steps else {}
        v = last_step_info.get("verifier", {})
        if v:
            kind = v.get("verifier", "?")
            st.write(f"**Type:** `{kind}`")
            if kind == "assertion_list":
                passed = v.get("passed", []); failed = v.get("failed", [])
                cols = st.columns(2)
                with cols[0]:
                    st.success(f"Passed ({len(passed)})")
                    for n in passed: st.write(f"- {n}")
                with cols[1]:
                    if failed:
                        st.error(f"Failed ({len(failed)})")
                        for n in failed: st.write(f"- {n}")
                    else:
                        st.info("No failed assertions")
            elif kind == "metric_threshold":
                st.metric(label=v.get("metric", "metric"),
                          value=f"{v.get('value', 0):.3f}",
                          delta=f"threshold {v.get('threshold', '?')}",
                          delta_color="off")
            elif kind == "composite":
                for child in v.get("children", []):
                    with st.expander(f"child: {child.get('verifier', '?')}"):
                        st.json(child)
            else:
                st.json(v)
        else:
            st.write("No verifier info recorded.")

        # KPIs (specific to ride-sharing tasks).
        last_kpis = None
        for s in reversed(traj.steps):
            if s.observation.data and "kpis" in s.observation.data:
                last_kpis = s.observation.data["kpis"]
                break
        if not last_kpis and last_step_info.get("verifier", {}).get("info", {}).get("kpis"):
            last_kpis = last_step_info["verifier"]["info"]["kpis"]
        if last_kpis:
            st.markdown("### Final KPIs")
            cols = st.columns(4)
            cols[0].metric("Trips seen", last_kpis.get("n_trips_seen", 0))
            cols[1].metric("Mean wait (min)",
                            f"{last_kpis.get('mean_pickup_wait_minutes', 0):.2f}")
            cols[2].metric("Completion rate",
                            f"{last_kpis.get('completion_rate', 0):.0%}")
            cols[3].metric("Revenue", f"${last_kpis.get('revenue', 0):.2f}")

        # Episode summary.
        st.markdown("### Episode summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Success", "yes" if traj.success else "no")
        m2.metric("Final reward", f"{traj.final_reward:.2f}")
        m3.metric("Steps", len(traj.steps))
        m4.metric("Tokens (in/out)",
                   f"{traj.meta.get('total_tokens_in', 0)} / "
                   f"{traj.meta.get('total_tokens_out', 0)}")

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save_dir = RUNS_DIR / f"ui-{ts}-{agent_choice}"
        save_dir.mkdir(parents=True, exist_ok=True)
        traj_path = save_dir / f"{task_id.replace('/', '_')}__{seed}.jsonl"
        traj.write(traj_path)
        st.caption(f"Trajectory saved: `{traj_path.relative_to(_PROJ_ROOT)}`")
        st.download_button(
            "Download trajectory JSONL",
            data=traj.to_jsonl(),
            file_name=traj_path.name,
            mime="application/x-jsonlines",
        )

        env.close()


# --------------------------------------------------------------------------- #
# PAGE: Browse Runs
# --------------------------------------------------------------------------- #

def _read_scorecard(d: Path):
    p = d / "scorecard.json"
    if not p.exists(): return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError: return None


def _read_index(d: Path) -> list[dict]:
    p = d / "index.jsonl"
    if not p.exists(): return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out


def page_browse_runs():
    st.title("Browse runs")
    if not RUNS_DIR.exists():
        st.info("No runs yet. Use **Run Episode** or `python eval/run.py ...`.")
        return
    runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()],
                   key=lambda d: d.stat().st_mtime, reverse=True)
    if not runs:
        st.info("No runs yet.")
        return
    rows = []
    for r in runs:
        sc = _read_scorecard(r) or {}
        ov = sc.get("overall", {})
        rows.append({
            "run": r.name,
            "agent": sc.get("agent", "?"),
            "model": sc.get("model") or "",
            "episodes": ov.get("n", "?"),
            "success_rate": (f"{ov['success_rate']:.0%}"
                              if "success_rate" in ov else "?"),
            "modified": datetime.fromtimestamp(r.stat().st_mtime)
                         .strftime("%Y-%m-%d %H:%M"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect", options=[r.name for r in runs])
    run_dir = RUNS_DIR / selected
    sc = _read_scorecard(run_dir) or {}
    if sc:
        st.subheader("Scorecard")
        ov = sc.get("overall", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Episodes", ov.get("n", 0))
        c2.metric("Success",
                   f"{ov.get('success_rate', 0):.0%}" if ov else "?")
        c3.metric("Reward", f"{ov.get('mean_reward', 0):.2f}")
        c4.metric("Tokens",
                   f"{ov.get('total_tokens_in', 0)} / "
                   f"{ov.get('total_tokens_out', 0)}")
        per_task = sc.get("tasks", {})
        if per_task:
            df = pd.DataFrame([
                {"task": tid, "n": m.get("n"),
                 "success_rate": f"{m.get('success_rate', 0):.0%}",
                 "mean_reward": f"{m.get('mean_reward', 0):.2f}",
                 "mean_steps": f"{m.get('mean_steps', 0):.1f}",
                 "errors": json.dumps(m.get("errors", {}))}
                for tid, m in per_task.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# PAGE: Task Catalog
# --------------------------------------------------------------------------- #

def page_task_catalog():
    st.title("Task catalog")
    factory = _shared_factory()
    for tid in all_task_ids():
        with st.expander(tid):
            task = TASK_REGISTRY[tid](seed=0)
            st.markdown(f"**Difficulty:** `{task.difficulty}`  •  "
                         f"**Max steps:** `{task.max_steps}`")
            try:
                env = GymEnvironment(task=task, sandbox_factory=factory)
                obs, info = env.reset()
                st.markdown("**Goal prompt the agent sees:**")
                st.markdown(f"> {obs.text}")
                st.markdown("**Initial state:**")
                st.json(obs.data)
                st.markdown("**Tools:**")
                for n in info["tools"]:
                    spec = task.tool_registry.get(n)
                    if spec:
                        st.markdown(f"- **`{n}`** — {spec.description}")
                env.close()
            except Exception as e:
                st.error(f"Preview failed: {e}")


# --------------------------------------------------------------------------- #
# PAGE: Live World
# --------------------------------------------------------------------------- #

def page_live_world():
    st.title("Live world state")
    factory = _shared_factory()
    if "ui_world_sandbox" not in st.session_state:
        st.session_state.ui_world_sandbox = factory()
    sb = st.session_state.ui_world_sandbox

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Reset tenant", use_container_width=True):
        sb.reset()
        st.success("Reset.")
    if c2.button("Seed (50d / 100r)", use_container_width=True):
        sb.rs.seed(n_drivers=50, n_riders=100, seed=0)
        st.success("Seeded.")
    if c3.button("Tick +60s", use_container_width=True):
        sb.rs.tick(60.0)
    if c4.button("New tenant", use_container_width=True):
        st.session_state.ui_world_sandbox = factory()
        st.rerun()

    snap = sb.snapshot()
    counts = {
        "drivers": len(snap["drivers"]),
        "riders": len(snap["riders"]),
        "trips": len(snap["trips"]),
        "refunds": len(snap["refunds"]),
        "disputes": len(snap["disputes"]),
        "incidents": len(snap["incidents"]),
        "lost_items": len(snap["lost_items"]),
        "events": len(snap["events"]),
        "sent_messages": len(snap["sent_messages"]),
    }
    st.dataframe(pd.DataFrame([counts]), use_container_width=True,
                 hide_index=True)

    st.markdown(f"**KPIs** (now = {snap['now']:.0f}s)")
    st.json(snap.get("kpis", {}))

    resource = st.selectbox("Inspect", list(counts.keys()))
    data = snap.get(resource, {})
    rows = list(data.values()) if isinstance(data, dict) else data
    if rows:
        try:
            st.dataframe(pd.DataFrame(rows[:200]), use_container_width=True,
                         hide_index=True)
        except Exception:
            st.json(rows[:50])
    else:
        st.info("Empty.")


# --------------------------------------------------------------------------- #
# PAGE: Map View
# --------------------------------------------------------------------------- #

def page_map_view():
    st.title("Map view")
    st.caption("Live 2D map of the city. Drivers are status-coloured; "
                "pending requests are cyan diamonds; zones colour by surge multiplier.")
    factory = _shared_factory()
    if "ui_map_sandbox" not in st.session_state:
        st.session_state.ui_map_sandbox = factory()
    sb = st.session_state.ui_map_sandbox

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Reset", use_container_width=True):
        sb.reset()
    if c2.button("Seed (50d / 30r)", use_container_width=True):
        sb.rs.seed(n_drivers=50, n_riders=30, seed=0)
    dt = c3.selectbox("Tick size", [30.0, 60.0, 120.0, 300.0], index=1)
    if c4.button(f"Tick +{int(dt)}s", use_container_width=True):
        sb.rs.tick(dt)

    auto = st.checkbox("Auto-advance (1 tick / 1.5s)", value=False)

    from rideshare_gym.mock_server.store import get_world
    world = get_world(sb.tenant_id)
    img = render_map(world, title=world.city.name)
    st.image(img, use_container_width=True)

    if auto:
        time.sleep(1.5)
        sb.rs.tick(dt)
        st.rerun()


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

if page == "Run Episode":
    page_run_episode()
elif page == "Browse Runs":
    page_browse_runs()
elif page == "Task Catalog":
    page_task_catalog()
elif page == "Live World":
    page_live_world()
elif page == "Map View":
    page_map_view()
