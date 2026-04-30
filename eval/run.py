"""Eval harness CLI.

Usage:
    python eval/run.py --tasks rideshare/match_single_ride --agent gold_oracle --n-episodes 3
    python eval/run.py --tasks "rideshare/*" --agent claude_baseline --n-episodes 5 --parallel 2 --adversarial latency,rate_limit
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow running as `python eval/run.py` (cwd not on sys.path otherwise).
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from typing import Any

from rideshare_gym.core.adversarial import FixtureMutator, Perturbation
from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.core.recorder import Trajectory, write_run_index
from rideshare_gym.rideshare_sandbox import (
    in_process_sandbox_factory,
    remote_sandbox_factory,
)
from rideshare_gym.tasks import REGISTRY as TASK_REGISTRY
from rideshare_gym.tasks import all_task_ids

from eval.scorecard import make_scorecard

DEFAULT_PERTURBATIONS = {
    "latency": {"endpoint": "*", "p": 0.3, "ms": 200},
    "rate_limit": {"p": 0.05, "status": 429, "retry_after": 1},
    "webhook_drop": {"event": "*", "drop_rate": 0.2},
    "stale_get": {"endpoint": "/orders/*", "p": 0.05, "lag_ms": 1000},
    "partial_failure": {"action": "refund", "step": "restock_inventory", "p": 0.10},
}


def _resolve_tasks(patterns: list[str]) -> list[str]:
    """Expand glob patterns against the task registry."""
    out: list[str] = []
    for pat in patterns:
        if pat in TASK_REGISTRY:
            out.append(pat)
        else:
            matched = [t for t in all_task_ids() if fnmatch.fnmatch(t, pat)]
            if not matched:
                raise SystemExit(f"no tasks match {pat!r}; known: {sorted(all_task_ids())}")
            out.extend(matched)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _build_perturbations(spec: str | None) -> list[Perturbation]:
    if not spec:
        return []
    out: list[Perturbation] = []
    for kind in (s.strip() for s in spec.split(",")):
        if not kind:
            continue
        if kind not in DEFAULT_PERTURBATIONS:
            raise SystemExit(f"unknown perturbation kind: {kind!r}")
        out.append(Perturbation(kind=kind, params=DEFAULT_PERTURBATIONS[kind]))
    return out


def _build_agent(
    agent_name: str, *,
    model: str | None,
    base_url: str | None,
    verbose: bool,
    litellm_tags: list[str] | None,
):
    if agent_name == "gold_oracle":
        from agents.gold_oracle import GoldOracleAgent
        return GoldOracleAgent()
    if agent_name == "claude_baseline":
        from agents.claude_baseline import ClaudeBaselineAgent
        kwargs = {"verbose": verbose}
        if model:
            kwargs["model"] = model
        return ClaudeBaselineAgent(**kwargs)
    if agent_name == "litellm":
        from agents.litellm_agent import LiteLLMAgent
        kwargs: dict[str, Any] = {"verbose": verbose}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        if litellm_tags:
            kwargs["litellm_tags"] = litellm_tags
        return LiteLLMAgent(**kwargs)
    raise SystemExit(f"unknown agent: {agent_name!r}")


def run_one_episode(
    task_id: str, *,
    seed: int,
    sandbox_factory,
    agent,
    perturbations: list[Perturbation],
) -> Trajectory:
    base_task = TASK_REGISTRY[task_id](seed=seed)
    task = FixtureMutator(base_task, perturbations) if perturbations else base_task
    env = GymEnvironment(task=task, sandbox_factory=sandbox_factory)
    try:
        traj = agent.run(env)
    finally:
        env.close()
    return traj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True,
                     help="Comma-separated task ids or glob patterns (e.g. 'rideshare/*').")
    ap.add_argument("--agent", default="gold_oracle",
                     choices=["gold_oracle", "claude_baseline", "litellm"])
    ap.add_argument("--model", default=None,
                     help="Model id. claude_baseline: claude-opus-4-7. "
                          "litellm: openai/gpt-4o, openai/gpt-5, "
                          "us.anthropic.claude-3-7-sonnet-20250219-v1:0, etc.")
    ap.add_argument("--base-url", default=None,
                     help="OpenAI-compatible endpoint URL "
                          "(litellm agent only; default https://llm-west.ncsu-las.net/v1).")
    ap.add_argument("--litellm-tags", default=None,
                     help="Comma-separated tags sent via x-litellm-tags header.")
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adversarial", default="",
                     help="Comma-separated perturbation kinds (latency, rate_limit, webhook_drop, stale_get, partial_failure).")
    ap.add_argument("--mock-base-url", default=None,
                     help="If set, talk to a remote rideshare-mock instead of in-process.")
    ap.add_argument("--out", default=None,
                     help="Output dir; default runs/<timestamp>-<agent>/")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    task_ids = _resolve_tasks(args.tasks.split(","))
    perturbations = _build_perturbations(args.adversarial)
    tags = [t.strip() for t in args.litellm_tags.split(",")] if args.litellm_tags else None
    agent = _build_agent(
        args.agent, model=args.model, base_url=args.base_url,
        verbose=args.verbose, litellm_tags=tags,
    )

    if args.mock_base_url:
        sandbox_factory = remote_sandbox_factory(base_url=args.mock_base_url)
    else:
        sandbox_factory = in_process_sandbox_factory(tenant_prefix="eval")

    out_dir = Path(args.out) if args.out else Path(
        f"runs/{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}-{args.agent}-{uuid.uuid4().hex[:6]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    print(f"Running {len(task_ids)} tasks x {args.n_episodes} episodes "
          f"= {len(task_ids) * args.n_episodes} eps "
          f"(parallel={args.parallel}, agent={args.agent}, "
          f"adversarial={args.adversarial or 'off'})")
    print(f"Output: {out_dir}")

    work: list[tuple[str, int]] = []
    for tid in task_ids:
        for i in range(args.n_episodes):
            work.append((tid, args.seed + i))

    trajectories: list[Trajectory] = []

    def _do_one(item: tuple[str, int]) -> Trajectory:
        tid, seed = item
        return run_one_episode(
            tid, seed=seed, sandbox_factory=sandbox_factory,
            agent=agent, perturbations=perturbations)

    if args.parallel <= 1:
        for item in work:
            t = _do_one(item)
            trajectories.append(t)
            t.write(traj_dir / f"{t.task_id.replace('/', '_')}__{t.seed}__{t.episode_id[:8]}.jsonl")
            print(f"  {t.task_id} seed={t.seed} -> success={t.success} "
                  f"reward={t.final_reward:.2f} steps={len(t.steps)} "
                  f"err={t.error_category}")
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(_do_one, item): item for item in work}
            for fut in as_completed(futures):
                t = fut.result()
                trajectories.append(t)
                t.write(traj_dir / f"{t.task_id.replace('/', '_')}__{t.seed}__{t.episode_id[:8]}.jsonl")
                print(f"  {t.task_id} seed={t.seed} -> success={t.success} "
                      f"reward={t.final_reward:.2f} steps={len(t.steps)} "
                      f"err={t.error_category}")

    write_run_index(out_dir, trajectories)
    sc = make_scorecard(trajectories, agent_id=args.agent, model=args.model)
    scorecard_path = out_dir / "scorecard.json"
    scorecard_path.write_text(json.dumps(sc, indent=2, default=str), encoding="utf-8")

    print()
    print(f"=== Scorecard ({sc['overall']['n']} episodes) ===")
    print(f"Overall success: {sc['overall']['success_rate']:.1%}  "
          f"mean reward: {sc['overall']['mean_reward']:.2f}  "
          f"errors: {sc['overall']['errors']}")
    for tid, m in sc["tasks"].items():
        print(f"  {tid}: success={m['success_rate']:.1%}  "
              f"reward={m['mean_reward']:.2f}  steps={m['mean_steps']:.1f}  "
              f"errors={m['errors']}")
    print(f"\nScorecard: {scorecard_path}")
    print(f"Trajectories: {traj_dir} ({len(trajectories)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
