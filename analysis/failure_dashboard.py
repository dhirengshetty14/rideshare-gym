"""Streamlit page for inspecting failure clusters from training rollouts.

Run with:
    streamlit run analysis/failure_dashboard.py -- --traj-dir runs/baseline-rollouts/trajectories

Shows: top failure clusters, per-task error breakdown, drill-down into
individual failure trajectories with the exact tool calls that didn't work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import pandas as pd
import streamlit as st

from training.data.failure_miner import (
    cluster_failures, cluster_to_dict, per_task_summary,
)
from training.data.trajectory_to_sft import load_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=Path,
                         default=Path("runs"))
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="Rideshare Gym — Failure Mining",
                        layout="wide")
    st.title("Failure mining")
    st.caption(f"Reading from `{args.traj_dir}`")

    # Choose a run dir.
    if args.traj_dir.is_dir() and (args.traj_dir.name == "trajectories"
                                     or any((args.traj_dir / "*.jsonl").parent.glob("*.jsonl"))):
        traj_dir = args.traj_dir
    else:
        runs = sorted([d for d in args.traj_dir.iterdir() if d.is_dir()],
                       key=lambda d: d.stat().st_mtime, reverse=True)
        run_names = [r.name for r in runs]
        if not run_names:
            st.error("No runs found.")
            return
        chosen = st.sidebar.selectbox("Run", run_names)
        traj_dir = (args.traj_dir / chosen / "trajectories"
                     if (args.traj_dir / chosen / "trajectories").exists()
                     else args.traj_dir / chosen)

    trajs = load_trajectories(traj_dir)
    st.write(f"Loaded **{len(trajs)}** trajectories")

    # Per-task summary.
    st.subheader("Per-task summary")
    summary = per_task_summary(trajs)
    rows = []
    for tid, m in summary.items():
        rows.append({
            "task": tid,
            "n_episodes": m["n_episodes"],
            "success_rate": f"{m['success_rate']:.0%}",
            **{f"err::{k}": v for k, v in m["error_breakdown"].items()},
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Failure clusters.
    st.subheader("Top failure clusters")
    clusters = cluster_failures(trajs)
    cluster_rows = []
    for c in clusters[:50]:
        cluster_rows.append({
            "count": c.count,
            "task": c.task_id,
            "last_tool": c.last_tool,
            "first_failed_assertion": c.first_failed_assertion,
            "error_category": c.error_category,
        })
    if cluster_rows:
        st.dataframe(pd.DataFrame(cluster_rows),
                     use_container_width=True, hide_index=True)

    # Drill-down.
    st.subheader("Inspect a cluster")
    if clusters:
        labels = [
            f"#{i} {c.task_id} | {c.last_tool} | {c.first_failed_assertion} (n={c.count})"
            for i, c in enumerate(clusters[:50])
        ]
        idx = st.selectbox("Cluster", range(len(labels)),
                            format_func=lambda i: labels[i])
        c = clusters[idx]
        st.json(cluster_to_dict(c, with_traj_summaries=True))
        st.markdown("**Example trajectories:**")
        for t in c.example_trajectories[:3]:
            with st.expander(f"{t.episode_id[:8]} — {len(t.steps)} steps, "
                              f"reward={t.final_reward:.2f}"):
                step_rows = []
                for s in t.steps:
                    info = s.info or {}
                    v = info.get("verifier", {})
                    step_rows.append({
                        "#": s.index,
                        "tool": s.action.name,
                        "args": json.dumps(s.action.arguments)[:150],
                        "ok": "yes" if info.get("tool_ok") else "no",
                        "reward": s.reward,
                        "passed": ", ".join(v.get("passed", [])),
                        "failed": ", ".join(v.get("failed", [])),
                    })
                if step_rows:
                    st.dataframe(pd.DataFrame(step_rows),
                                  use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
