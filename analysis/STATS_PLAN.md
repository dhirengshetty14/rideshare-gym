# Statistical evaluation plan — before vs after training

This document defines exactly what we measure on Qwen2.5-7B-Instruct
**before** training (the baseline), **after** each training stage (SFT, DPO,
GRPO), and how we compare them.

The goal is to answer ONE precise question:

> Did the rideshare-gym training pipeline actually improve the model, and by
> how much, and on which tasks?

Without statistics, "70% → 85%" is anecdote. With statistics — paired tests,
confidence intervals, effect sizes — it becomes a defensible claim.

---

## 1. Sampling design

### Sample size
- **N = 50 episodes per task per stage.**
  - 12 tasks × 50 episodes = **600 episodes per stage**
  - 4 stages (baseline, SFT, DPO, GRPO) = **2,400 episodes total**

### Why 50?
- Standard error of a proportion estimate at p=0.5 with N=50 is ~7 percentage points.
- Detects an absolute improvement of 10 pp with ≥80% statistical power.
- Total wall-clock budget on 1× A100: ~3-6 h per stage. Manageable.

### Paired design — same seeds across stages
Every stage uses **seeds 0..49** for each task. This is critical: it lets us
do *paired* statistical tests (McNemar's test on success, paired Wilcoxon on
reward) which are far more powerful than unpaired tests.

### Eval temperature
- **Temperature = 0.0** (greedy decoding) for stats runs. Removes sampling
  noise from the comparison. Trained vs baseline on identical inputs gives
  identical "what would the model do?" answers.

(We also collect rollouts at temperature=0.7 elsewhere in the pipeline for
training data construction — that's a separate phase.)

---

## 2. Per-trajectory statistics

For each of the 600 trajectories per stage, we record:

| Field | Type | Source |
|---|---|---|
| `task_id` | string | trajectory.task_id |
| `seed` | int | trajectory.seed |
| `success` | bool | trajectory.success |
| `final_reward` | float in [0, 1] | trajectory.final_reward |
| `n_steps` | int | len(trajectory.steps) |
| `wall_time_seconds` | float | trajectory.meta.wall_time_seconds |
| `total_tokens_in` | int | trajectory.meta.total_tokens_in |
| `total_tokens_out` | int | trajectory.meta.total_tokens_out |
| `error_category` | enum or None | trajectory.error_category |
| `tool_calls_made` | int | count of steps |
| `tool_calls_ok` | int | sum(step.info["tool_ok"]) |
| `tool_calls_wrong_args` | int | sum("wrong_args" in step.info["tool_error"]) |
| `tool_calls_unknown_tool` | int | sum("unknown_tool" in step.info["tool_error"]) |
| `unique_tools_used` | int | len({step.action.name for step in steps}) |

---

## 3. Per-task statistics (aggregated across 50 seeds)

For each task, we compute:

| Stat | What it captures |
|---|---|
| `success_rate` | fraction of episodes that succeeded |
| `success_rate_ci_95` | bootstrap 95% CI on success rate (1000 resamples) |
| `mean_reward`, `std_reward`, `median_reward` | reward distribution |
| `mean_steps`, `median_steps` | efficiency |
| `mean_wall_time_s` | latency |
| `mean_tokens_in`, `mean_tokens_out` | cost proxies |
| `error_breakdown` | dict: `{goal_incomplete: n, wrong_tool: n, wrong_args: n, crashed: n}` |
| `tool_call_validity_rate` | tool_calls_ok / tool_calls_made |
| `wrong_args_rate` | tool_calls_wrong_args / tool_calls_made |
| `unknown_tool_rate` | tool_calls_unknown_tool / tool_calls_made |
| `mean_unique_tools` | tool diversity |

---

## 4. Overall statistics (aggregated across all 600 episodes per stage)

| Stat | What it tells us |
|---|---|
| `overall_success_rate` (with 95% CI) | "what fraction of all attempts succeeded?" |
| `mean_reward` (with 95% CI) | average partial credit |
| `success_rate_by_difficulty` (easy / medium / hard) | does training help equally across difficulties? |
| `dominant_failure_mode` | which error category is most common? |
| `total_tokens` | absolute compute spend (proxy for cost) |
| `mean_episode_wall_time_s` | latency profile |
| `tool_call_validity_rate` | "does the model emit syntactically-valid tool calls?" |

---

## 5. Before-vs-after comparison

For each pair of stages (baseline → trained, SFT → DPO, etc.) we compute:

### 5a. Paired success-rate test (McNemar's)

For every (task, seed) pair we have ONE outcome before training and ONE
after. Build the 2×2 contingency table:

```
                AFTER pass    AFTER fail
BEFORE pass         a            b
BEFORE fail         c            d
```

McNemar's test on `(b, c)` gives a p-value for "did the success rate change?"
This is a **paired** test — far more powerful than comparing pooled
proportions. Standard for before/after agent benchmarks.

### 5b. Per-task success-rate delta

`delta_per_task[t] = after_success_rate[t] - before_success_rate[t]`
With bootstrap 95% CI. We expect positive deltas on most tasks.

### 5c. Effect size (Cohen's h)

Cohen's h on proportion differences:
`h = 2 × (asin(sqrt(p_after)) - asin(sqrt(p_before)))`

Interpretation:
- |h| < 0.2 — small
- 0.2 ≤ |h| < 0.5 — medium
- |h| ≥ 0.5 — large

### 5d. Reward distribution comparison

- Paired Wilcoxon signed-rank test on per-(task,seed) reward differences
- Bootstrap CI on the mean reward delta
- KS-test on the reward distributions (sanity check)

### 5e. Failure-mode shift

Did the dominant failure category change after training? Did `wrong_args` go
down (model emits better-formatted tool calls)? Did `goal_incomplete` go
down (model finishes more episodes)?

We report a 4×2 table (4 error categories × 2 stages) and a chi-square test
of independence.

### 5f. Tool-call quality shift

- `wrong_args_rate` delta — did the model emit more syntactically-valid
  tool calls after training?
- `unknown_tool_rate` delta — did the model stop calling tools that don't
  exist?
- `mean_unique_tools` delta — does the model explore more tools or fewer?

These are LLM-specific quality metrics that surface improvements in
tool-using behaviour beyond raw success rate.

---

## 6. Output artefacts

After running the full pipeline, the following files are produced under
`analysis/`:

| File | What's in it |
|---|---|
| `baseline_stats.json` | All stats from §2-§4 for the baseline model |
| `sft_stats.json` | Same for the SFT-tuned checkpoint |
| `dpo_stats.json` | Same for the DPO-tuned checkpoint |
| `grpo_stats.json` | Same for the final GRPO checkpoint |
| `comparison_baseline_vs_grpo.json` | All §5 numbers for the headline comparison |
| `report.md` | Human-readable markdown report — the deliverable for the mentor |
| `training_curves.png` | One image: success rate per task at each stage |
| `confidence_intervals.png` | Per-task delta with 95% CI bars |
| `failure_shift.png` | Bar chart of error-category counts before vs after |

The `report.md` is the **single artefact** to share with the mentor. It
contains a one-line headline (e.g. "GRPO improves overall success from
38.4% to 67.1%, p < 0.001, Cohen's h = 0.59"), per-task deltas, and the
key plots.

---

## 7. Acceptance criteria

The training pipeline is "successful" if:

1. **Overall success rate increases by ≥15 pp** from baseline to GRPO,
   with p < 0.05 (paired McNemar) and Cohen's h ≥ 0.3 (medium effect).
2. **At least one hard task (H1, H2, H3) goes from <10% to ≥30% success.**
3. **Tool-call validity rate (% of calls that are syntactically valid)
   improves by ≥10 pp.** This isolates whether training fixed the
   "model emits malformed JSON" failure mode.
4. **No task regresses by more than 10 pp.** Catches catastrophic forgetting.

Failing to meet (1) means the pipeline doesn't work as expected — investigate
training hyperparameters or reward signal density.

Failing (2) but passing (1) is a partial win — it means SFT/DPO/GRPO improved
the easy and medium tasks but the hardest tasks are still beyond the model's
capability at this size.

---

## 8. Why this matters for the mentor's ask

> "The gym should improve the agent's performance. Show me the before/after."

The §5 comparison numbers are the literal answer. Specifically:

- §5a McNemar p-value answers "was the improvement statistically real?"
- §5b per-task delta answers "where exactly did the gym help?"
- §5c effect size answers "how much did it help?"
- §5e failure-mode shift answers "what specific weaknesses did the gym fix?"

The §7 acceptance criteria turn "did the gym work?" from a vibe into a
binary yes/no decision. That's the rigour your mentor is asking for.
