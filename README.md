# Rideshare Ops RL Gym

A real-time benchmark for **AI agents that run a ride-sharing platform** like Uber or Lyft. The agent plays the platform's brain: matching riders to drivers, setting surge pricing during demand spikes, recovering lost items, refunding chargeback disputes, catching coordinated fraud rings, escalating safety incidents, and rebalancing the fleet for predicted events. The gym scores every action automatically and produces training data for the next agent.

> **No real customer data, no real Uber backend, no real money.** Everything is synthetic and reproducible by seed. The shapes mirror real ride-sharing platforms; the fraud patterns and operational flows are the actual ones used in production systems.

---

## Table of contents

1. [What is this, in one paragraph](#what-is-this-in-one-paragraph)
2. [Why this exists](#why-this-exists)
3. [What "RL gym" means here](#what-rl-gym-means-here)
4. [The 12 scenarios, with examples](#the-12-scenarios-with-examples)
5. [The simulator under the hood](#the-simulator-under-the-hood)
6. [What an episode actually does](#what-an-episode-actually-does)
7. [Inputs and outputs at every level](#inputs-and-outputs-at-every-level)
8. [What data is in the gym](#what-data-is-in-the-gym)
9. [Adversarial mode](#adversarial-mode)
10. [Quick start](#quick-start)
11. [The Streamlit UI](#the-streamlit-ui)
12. [Project layout](#project-layout)
13. [Architecture diagram](#architecture-diagram)
14. [Running real LLMs](#running-real-llms)
15. [Sample scorecards](#sample-scorecards)
16. [How this could be used to actually train a model](#how-this-could-be-used-to-actually-train-a-model)
17. [What's not built (post-MVP)](#whats-not-built-post-mvp)

---

## What is this, in one paragraph

This repo is a **practice simulator** for AI agents that run a ride-sharing platform. The simulator is a 2D city of 6 zones with drivers physically moving on the map, riders requesting trips, demand fluctuating with a daily curve and special events (concert lets out, rush hour), and surge pricing responding to supply/demand imbalance. Time advances tick-by-tick. The agent (Claude, GPT, or our deterministic gold oracle baseline) is given a problem and a small set of tools — API calls like `match_ride`, `set_surge`, `freeze_account`, `escalate_incident`. It picks a tool, calls it, looks at the result, picks the next tool, and so on. After every action a **judge** (a Python verifier) checks whether the outcome matches success and gives the agent a reward between 0 and 1. When the judge says the goal is met, the episode ends. Everything is recorded — every action, every observation, every reward, every GPS waypoint, every dispatched trip, every fraud freeze — so you can replay episodes, compare agents, and use the data to train better ones.

That's the whole project.

---

## Why this exists

Loads of "AI for ride-sharing" startups exist — companies building agents to handle ops tasks like dispatch, fraud detection, customer support, and safety incident response. Nobody had a public benchmark to evaluate them. Existing agent benchmarks (τ-bench, AppWorld, WorkArena, our own [shopify-gym](https://github.com/dhirengshetty14/shopify-gym)) are either stateless or weakly stateful — none of them have a real-time multi-agent simulator with drivers physically moving on a map. This fills that gap.

---

## What "RL gym" means here

**Reinforcement Learning** is a broad family of techniques where an *agent* learns by interacting with an *environment*, taking actions, and receiving rewards. The picture is always the same:

```
Agent ──action──▶ Environment
  ▲                 │
  │ observation     │
  └─────────────────┘
  ◀─── reward ──────┘
```

An **RL gym** (the term comes from [Gymnasium](https://gymnasium.farama.org/)) is a standardised environment that any RL algorithm can plug into. Two methods define the contract:

- `reset()` — start a fresh episode, return the first observation
- `step(action)` — take one action, return `(observation, reward, terminated, truncated, info)`

That's what we built. The "environment" happens to be a simulated ride-sharing platform.

> **Are we training a model here?** Not yet. Right now we're doing **evaluation** — running episodes, scoring agents, comparing models. Model weights don't change. But every episode produces a *trajectory* file in exactly the format you'd need to fine-tune a model later. So the gym is doing two jobs at once: it's a benchmark *and* a data factory.

---

## The 12 scenarios, with examples

Each scenario is a self-contained operations problem. The agent gets a goal in plain English, a small toolbox, and has to figure out the rest by exploring the API.

### Easy (atomic, perfect-info — the agent has everything it needs)

#### E1: `match_single_ride`

**The story.** One pending ride request from downtown to the airport. Five idle drivers are spread across the city. Pick the closest-ETA one and dispatch.

**Tools:** `list_pending_requests`, `list_idle_drivers`, `match_ride`, `auto_match_nearest`, `get_trip`

**The judge checks:**
- Did the trip transition from `requested` → `matched`?
- Did the agent pick the actually-closest driver?
- Was a dispatch_log entry written with the ETA + alternatives_considered?

#### E2: `refund_cancelled_trip`

**The story.** A driver cancelled the trip after the rider had been waiting 12 minutes. The rider was charged. Issue a full refund and notify them.

**The judge checks:**
- Exactly one refund was created
- Refund amount equals the trip total fare
- Rider received a notification message

#### E3: `verify_driver_documents`

**The story.** A driver's license expired 5 days ago and they're blocked from picking up rides. Verify their documents.

**The judge checks:**
- `docs_verified` flips from `False` to `True`

### Medium (multi-step, decisions matter)

#### M1: `surge_demand_spike`

**The story.** A 5-minute concert-let-out event hits downtown. Demand surges 10x. Supply is unchanged. Raise the surge multiplier on downtown to balance demand, then dispatch as many drivers as possible.

**The judge checks (composite):**
- Surge in downtown raised to ≥ 1.5x within 60s of the spike
- Cumulative `unmatched_requests / total_requests` < 40% (metric threshold)

#### M2: `fraud_ring_detection`

**The story.** Recent chargebacks suggest an account-farm ring. 5 rider accounts share a device fingerprint and payment-card BIN. Automatic risk scoring flagged 3 of them as high-risk; 2 are subtle. Find ALL 5 and freeze each. Don't freeze any legit riders.

**The judge runs an F1 metric** with **false positives penalised twice as heavily as false negatives** (cancelling a legit account is a much worse error than missing a subtle fraud). F1 ≥ 0.9 to pass.

#### M3: `lost_item_recovery`

**The story.** A rider reports they left their wallet on a completed trip. The driver has done 3 more trips since.

**Required flow** (mirrors Uber's actual lost-item workflow):
1. Create a lost-item record
2. Assign it to the *original* driver (not whoever's nearest)
3. Schedule a return-pickup time and place
4. Both parties get a unique handoff code
5. Notify both via push

**The judge checks all 6 of these.**

#### M4: `driver_pay_dispute`

**The story.** A driver disputes their pay on a completed trip. They claim surge was 1.8x but they were paid as if 1.0x. Reconstruct the truth from the trip record + GPS log + surge zone log, recompute the correct payout, and adjust.

**The judge checks:**
- Payout adjustment made within $0.50 of the underpayment
- Driver notified via push

#### M5: `accident_incident_response`

**The story.** Mid-trip vehicle damage. Trip was cancelled. Severity is T2.

**Required actions:**
- Escalate incident to ≥ T2
- Contact emergency services
- Refund the rider in full
- Compensate the driver for the lost trip
- Notify both parties

#### M6: `account_takeover_response`

**The story.** A rider account is suddenly logging in at 3am from a new device with a foreign IP — wildly different from their normal 7am-10pm login pattern from a stable device. Probably a stolen-credential takeover.

**Required flow:**
1. Freeze the account immediately (stop the bleeding)
2. Review the login history
3. Message the real owner via the verified channel (template: `account.takeover_check`)
4. After positive verification (a stub flag in this gym), restore the account

### Hard (real-time / coordinated / time-pressured)

#### H1: `realtime_dispatch_window` (FLAGSHIP)

**The story.** Run 30 simulated minutes (60 ticks of 30 seconds). Demand varies across all 6 zones with a peak around minute 18 (rush_hour event). ~30 drivers, online_pct 85%. The agent must call `tick()` itself between actions to advance simulator time.

**The judge checks (composite, all must pass):**
- Mean pickup wait < 4.0 minutes
- Completion rate ≥ 0.5
- Cancellation rate < 0.3
- Episode actually ran the full 30 minutes

This is the long-horizon flagship task — typical episode is ~150-250 tool calls. Even our greedy gold oracle doesn't fully clear all 4 KPI thresholds; that gap is the headroom for stronger agents.

#### H2: `event_surge_planning`

**The story.** A 5-minute `concert_let_out` event will fire in the stadium zone at t=90 simulated seconds. 25 idle drivers are scattered across the suburbs. Pre-position 6+ drivers to stadium AND raise surge to ≥1.8x **before t=90s**.

**Predictive action under a hard deadline.** Every `tick()` call burns budget. Tests whether the agent can reason about future events.

#### H3: `coordinated_fraud_response`

**The story.** A fraud ring is in the middle of an attack:
- 5 ring rider accounts have already booked 12 trips
- 4 victim drivers got chargeback-overcharged on past trips with this ring
- Every 30s another fraud trip will fire if nothing happens

**Required actions:**
- Freeze ALL 5 ring rider accounts
- Ban the shared device fingerprint
- Adjust each of the 4 victim drivers' payouts upward by the right amount (per ground truth)
- **Don't freeze any legit accounts** (FP penalty)

---

## The simulator under the hood

This is the new part vs. the [shopify-gym](https://github.com/dhirengshetty14/shopify-gym) (which was stateless API ops). Here we have a real **finite-state simulator** with a global clock.

### City

A 30 km × 30 km flat plane with 6 zones — `downtown`, `airport`, `university`, `stadium`, `suburb_n`, `suburb_s`. Travel times use Manhattan distance with per-zone traffic factors and a baseline 30 km/h speed. Stylized, but realistic enough that the agent can't game it through the API.

### Drivers

Each driver has a `Driver` record with:
- `location` (km coordinates)
- `status` (offline / idle / dispatched / picking_up / in_trip / break)
- `vehicle_type` (uberx / uberxl / uberblack)
- `rating`, `device_fingerprint`, `home_zone_id`, `docs_verified`, `flags`
- `cumulative_earnings_today`, `completed_trips_today`, `cancellation_count_today`

When dispatched, drivers physically move toward their `target_location` each tick at `speed_kmh`. When they arrive at the pickup, the trip transitions to `driver_arrived`. After a 30-second dwell, pickup happens and the driver heads for the dropoff.

### Trips

Trip lifecycle:

```
REQUESTED ──match()──▶ MATCHED ──driver_arrives()──▶ DRIVER_ARRIVED
                                                       │
                                                  rider_pickup()
                                                       ▼
                                                    IN_TRIP ──complete()──▶ COMPLETED
                                                       │
                                                       └──cancel()──▶ CANCELLED
```

Each trip records: requested_at, matched_at, picked_up_at, completed_at, cancellation reason, GPS waypoints (sampled per tick), surge multiplier at request time, full fare breakdown, ratings, refunds, disputes, lost items, incidents.

### Pricing — modelled on real Uber rate cards

| Vehicle | Base | Per km | Per min | Booking fee | Min fare |
|---|---|---|---|---|---|
| UberX | $2.55 | $0.95 | $0.34 | $2.50 | $4.00 |
| UberXL | $3.85 | $1.45 | $0.45 | $2.50 | $4.00 |
| UberBlack | $7.00 | $2.55 | $0.50 | $4.50 | $15.00 |

All multiplied by the current surge (clipped to [1.0, 5.0], rounded to 0.25 increments). 25% platform service fee. 8% tax.

Driver payout = `(distance_fare + time_fare) × surge × (1 - service_fee_pct)`.

### Surge dynamics

Each tick, per zone: `ratio = max(demand / max(supply, 1), 1)` →  surge = `clip(round_to_0.25(ratio), 1.0, 5.0)`. Hysteresis applied (no thrashing — moves at most one 0.25 step per tick). Agent can override via `set_surge(zone_id, multiplier, ttl_minutes)`.

### Demand

Poisson arrivals per zone, with a diurnal time-of-day multiplier (twin Gaussians at 8am and 6pm) plus event-driven multipliers. Each new request has an origin zone (Poisson-sampled) and a destination zone (sampled from an OD matrix biased toward different zones).

### Events

```python
WorldEvent(
    kind="concert_let_out",      # or traffic_jam, accident, weather, rush_hour, ...
    started_at=90.0,
    duration_seconds=300.0,
    affected_zones=["stadium"],
    severity=1.0,                # 0..1
)
```

Tasks plant events at setup (e.g. `event_surge_planning` plants the concert-let-out). Adversarial mode can layer additional events (random `traffic_event`).

### Fraud engine — three real-world archetypes

1. **Account farm chargeback ring** — N rider accounts share a payment-method fingerprint and BIN range; obvious_count of them carry visible high-risk flags, the rest are subtle.
2. **Collusion ring** — same operator runs multiple driver + rider accounts (shared fingerprint), books fake trips to extract surge incentives.
3. **GPS spoofing driver fraud** — driver's GPS log shows static or impossible-velocity locations, but trip "completed" successfully.

The agent's job in fraud tasks is to identify the cluster from these signals.

---

## What an episode actually does

Walking through the `match_single_ride` episode end-to-end.

### Setup

The gym builds a fresh tenant in 50 ms:
- 5 idle drivers planted at the centroids of 5 different zones
- 1 rider + 1 pending trip from `downtown` to `airport`
- A **ground truth** computed silently — which driver is actually closest by ETA. Stashed where the agent cannot read it.

### Goal prompt

The agent receives:

> One pending ride: trip 1007 from downtown to airport. Five idle drivers are available across zones. Match the trip to the BEST (closest-ETA) driver.

Tools: `list_pending_requests`, `list_idle_drivers`, `match_ride`, `auto_match_nearest`, `get_trip`.

### The agent's loop

**Step 0 — one inference (one call to GPT-4o or Claude):**
- Input: goal + tool definitions
- Model emits: `auto_match_nearest({"trip_id": 1007})`
- Mock server: identifies driver 1001 (downtown centroid is the pickup, driver 1001 is at downtown's centroid → ETA 0). Updates trip status to `MATCHED`, driver to `DISPATCHED`.
- Verifier runs: 3/3 assertions pass → reward 1.0, terminated=True.

**Total: 1 inference, ~30 ms wall-clock with the gold oracle, ~2-3 seconds with GPT-4o.**

For real-time tasks (H1), an episode is hundreds of inferences spread over 60-90 seconds wall clock as the agent calls `tick()` repeatedly to advance simulator time.

---

## Inputs and outputs at every level

| Level | Input | Output |
|---|---|---|
| **Whole gym** | task name, seed, agent, model, # episodes, perturbations | `scorecard.json` (success rate, mean reward, KPIs, error breakdown) + saved trajectory files |
| **Episode** | task setup + agent | one trajectory: list of (action, observation, reward) tuples + final success/fail |
| **Step** (one round) | a tool call from the agent | new observation + reward + terminated flag + KPIs |
| **One LLM inference** | system prompt + tool definitions + conversation history | a tool call (name + JSON arguments) |
| **One tool call** | tool name + arguments | API response (driver locations, trip details, GPS log, surge multipliers, …) |
| **Verifier** | the simulator's state after the action | pass/fail per assertion → reward 0 to 1 |

The model **weights are never changed**. We are *using* the model, not training it. Each episode is fresh; the model has no memory across episodes.

---

## What data is in the gym

**Everything is synthetic.** Generated freshly the moment each episode starts.

| What | Source |
|---|---|
| Customer (rider) names, emails, phones | Faker |
| Driver names, ratings, vehicle assignments | Faker |
| Pickup/dropoff coordinates | Sampled inside the relevant zone polygons |
| Trip records | Built by the simulator from request → match → complete state machine |
| GPS waypoints | Sampled per tick from each driver's actual position |
| Fraud ring fingerprints + BINs | `fraud_engine.py` — patterns based on published descriptions of real ride-share fraud rings |
| Surge multipliers | Computed live from per-zone supply/demand |
| Fares | Computed live from the rate cards above |
| Events | Planted by tasks (`concert_let_out` at t=90s, etc.) |

The same `seed=N` always produces the same fake world.

**No real PII, no real money, no real driver/rider data.** But the API contract, fraud signal patterns, and operational flows mirror real platforms (Uber, Lyft).

---

## Adversarial mode

Real ride-sharing platforms aren't a perfect world. APIs return 429, GPS drops, payments require 3DS challenges, drivers go offline mid-trip, traffic jams hit. We injected each of these as a flag you can turn on:

| Perturbation | What it does |
|---|---|
| `latency` | 30% of API calls take 200 ms longer |
| `rate_limit` | 5% of calls return HTTP 429 with `Retry-After` |
| `partial_failure` | Multi-step writes silently skip the secondary step (e.g. refund issued but driver-side message dropped) |
| `system_outage_partial` | One router family (e.g. `/api/v1/safety/`) returns 503 for the duration |
| `gps_dropout` | GPS log endpoint returns last-known-good for some duration |
| `payment_3ds` | Refunds randomly require a follow-up `confirm_3ds` step |
| `driver_offline_mid_trip` | Random in-flight trips drop their driver (status flips, agent must reassign) |
| `eta_variance` | Pricing/quote endpoints return ±25% noisy ETAs |
| `traffic_event` | Random `traffic_jam` event injected mid-episode in a zone |
| `fraud_pattern_drift` | Fraud ring fingerprints rotate mid-episode |
| `messaging_delay` | `send_to_*` returns 200 but the message lands minutes later |

Each perturbation is **seeded** for reproducibility. The success-rate gap between clean and adversarial mode is the benchmark's main signal — a robust agent retries 429s, verifies state after every multi-step write, and copes with partial information.

---

## Quick start

### Install

```bash
git clone https://github.com/dhirengshetty14/rideshare-gym.git
cd rideshare-gym

python -m venv .venv

# Activate (pick the one for your shell)
.venv\Scripts\Activate.ps1     # PowerShell (Windows Terminal default)
.venv\Scripts\activate.bat     # CMD
source .venv/Scripts/activate  # Git Bash / Linux / Mac

pip install -e ".[dev]"
```

### Run all 57 tests (no API key needed)

```bash
pytest tests/ -q
```

You should see `57 passed in ~4s`.

### Run the gold oracle on every task

```bash
python eval/run.py --tasks "rideshare/*" --agent gold_oracle --n-episodes 3
```

Expected: 10/12 tasks at 100% success. The 2 real-time tasks (H1, M1) deliberately leave headroom even for the optimal greedy oracle.

### Add chaos

```bash
python eval/run.py --tasks "rideshare/*" --agent gold_oracle --n-episodes 3 --adversarial latency,rate_limit,partial_failure,system_outage_partial
```

Watch success drop from 83% to ~60-75% — that's the benchmark signal.

---

## The Streamlit UI

```bash
streamlit run ui/app.py
```

Opens a browser tab at `http://localhost:8501` with **5 pages**:

1. **Run Episode** — pick task, agent, model, perturbations; click *Start episode*; watch the trajectory stream in row by row as each tool call completes. See the verifier breakdown, ride-sharing KPIs (mean wait, completion rate, revenue), and episode summary.
2. **Browse Runs** — every CLI/UI run is listed; drill into one to see the scorecard and per-task breakdown.
3. **Task Catalog** — what each of the 12 tasks does, the goal prompt, the tools, the initial state.
4. **Live World** — spin up a fresh tenant, seed it, tick time, and inspect every resource (drivers, riders, trips, refunds, disputes, incidents, lost items, events, sent messages).
5. **Map View** — **the killer demo view.** Matplotlib 2D map of the city showing:
   - Drivers as status-coloured dots (idle = green, dispatched = orange, picking up = blue, in trip = red, offline = gray)
   - Pending ride requests as cyan diamonds
   - Zones colour-shaded by current surge multiplier (gray = 1x, yellow = 1.25-1.5x, orange = 1.5-2x, red = 2x+)
   - Auto-advance checkbox ticks every 1.5 seconds for live demos

---

## Project layout

```
rideshare-gym/
├── pyproject.toml                            # uv workspace; pins deps
├── README.md                                 # this file
├── docker-compose.yml                        # mock + Toxiproxy for HTTP-mode eval
├── docker/
│   ├── rideshare-mock/Dockerfile
│   └── toxiproxy/toxiproxy.json
├── src/rideshare_gym/
│   ├── core/                                 # domain-agnostic gym scaffolding
│   │   ├── env.py                            # Gymnasium-compatible env
│   │   ├── task.py                           # AbstractTask
│   │   ├── verifier.py                       # 4 verifier patterns
│   │   ├── tools.py                          # ToolSpec, ToolRegistry
│   │   ├── sandbox.py                        # Sandbox protocol
│   │   ├── recorder.py                       # Trajectory + JSONL writer
│   │   ├── adversarial.py                    # Perturbation, FixtureMutator
│   │   └── types.py                          # ToolCall, ToolResult, Observation
│   ├── world/                                # ★ THE SIMULATOR ★
│   │   ├── city.py                           # 6-zone Bayview, Manhattan distances
│   │   ├── clock.py                          # SimClock — discrete time stepping
│   │   ├── drivers.py                        # state machine + movement physics
│   │   ├── riders.py                         # rider model (BIN, fingerprint)
│   │   ├── trips.py                          # full lifecycle + GPS log
│   │   ├── pricing.py                        # real Uber-style rate cards
│   │   ├── surge.py                          # supply/demand-driven, hysteresis
│   │   ├── demand.py                         # Poisson arrivals + diurnal pattern
│   │   ├── events.py                         # traffic, weather, concert_let_out
│   │   ├── fraud_engine.py                   # 3 real-world fraud archetypes
│   │   ├── matching.py                       # nearest-ETA helpers
│   │   └── world.py                          # World — composes everything
│   ├── mock_server/                          # FastAPI app exposing the world
│   │   ├── app.py
│   │   ├── store.py                          # per-tenant World registry
│   │   ├── perturbations.py                  # ride-sharing-aware middleware
│   │   ├── seed.py                           # Faker driver/rider populations
│   │   └── routers/                          # 12 routers
│   │       ├── marketplace.py                # match, set_surge, rebalance, ...
│   │       ├── trips.py                      # list/get/gps_log
│   │       ├── refunds.py
│   │       ├── disputes.py
│   │       ├── safety.py                     # T1/T2/T3 escalation
│   │       ├── fraud.py                      # freeze, ban, cluster
│   │       ├── lost_items.py                 # full Uber-style flow
│   │       ├── drivers.py                    # docs, payout, performance
│   │       ├── riders.py                     # account, login_history
│   │       ├── pricing.py                    # quote, surge zones
│   │       ├── messaging.py                  # push/SMS captured for verifiers
│   │       └── admin.py                      # reset, snapshot, perturb, tick, seed
│   ├── tasks/                                # the 12 MVP tasks
│   │   ├── _base.py
│   │   ├── match_single_ride.py              # E1
│   │   ├── refund_cancelled_trip.py          # E2
│   │   ├── verify_driver_documents.py        # E3
│   │   ├── surge_demand_spike.py             # M1
│   │   ├── fraud_ring_detection.py           # M2
│   │   ├── lost_item_recovery.py             # M3
│   │   ├── driver_pay_dispute.py             # M4
│   │   ├── accident_incident_response.py     # M5
│   │   ├── account_takeover_response.py      # M6
│   │   ├── realtime_dispatch_window.py       # H1 — flagship
│   │   ├── event_surge_planning.py           # H2
│   │   └── coordinated_fraud_response.py     # H3
│   ├── tools.py                              # 46 ToolSpec entries
│   └── rideshare_sandbox.py                  # ShopifySandbox-equivalent
├── agents/
│   ├── gold_oracle.py                        # solves all 12 (10/12 to 100%)
│   ├── claude_baseline.py                    # Anthropic SDK
│   ├── litellm_agent.py                      # OpenAI-compat (LAS LiteLLM, etc.)
│   └── prompts/rideshare_system.md           # platform-brain principles
├── eval/
│   ├── run.py                                # CLI parallel episode runner
│   ├── scorecard.py                          # per-task + overall metrics
│   ├── error_taxonomy.py                     # τ-bench-style error buckets
│   └── list_models.py                        # introspect any /v1/models endpoint
├── ui/
│   └── app.py                                # Streamlit, 5 pages incl. map view
└── tests/                                    # 57 tests, ~4s
    ├── unit/                                 # 27 — gym scaffolding (ports from shopify-gym)
    ├── world/                                # 18 NEW — simulator correctness
    ├── integration/                          # 6 — mock server end-to-end
    └── tasks/                                # 12 — gold oracle per-task
```

---

## Architecture diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI / CLI                           │
└─────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │  eval/run.py / ui/app.py    │  picks the agent + tasks + perts
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │   Agent (one of)            │
        │   - gold_oracle             │   one inference per step:
        │   - claude_baseline         │     in: history + tools
        │   - litellm                 │     out: a tool call
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │     GymEnvironment          │   Gymnasium API: reset() / step()
        │  (Gymnasium-compatible)     │
        └──────┬─────────────┬────────┘
               │             │
   reads tools │             │ runs after every action
               ▼             ▼
        ┌──────────┐  ┌──────────────┐
        │   Task   │  │  Verifier    │  pass/fail per assertion
        │ -tools() │  │              │  → reward 0..1
        │ -setup() │  └──────────────┘
        │-verifier │
        └────┬─────┘
             │ tools execute against
             ▼
        ┌─────────────────────────────┐
        │   RideshareSandbox          │   HTTP client (httpx / TestClient)
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌────────────────────────────────┐
        │ Mock FastAPI app + 12 routers  │ Per-tenant World instances
        │ - marketplace / trips          │ PerturbationMiddleware
        │ - refunds / disputes           │ Tick endpoint advances clock
        │ - safety / fraud               │
        │ - lost_items / messaging       │
        │ - drivers / riders / pricing   │
        │ - admin (reset/perturb/seed)   │
        └──────────────┬─────────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │   World (the simulator)     │   ★ NEW vs shopify-gym ★
        │  - SimClock                 │
        │  - City (6 zones, distance) │
        │  - Drivers (state machine + │
        │      movement physics)      │
        │  - Trips (lifecycle + GPS)  │
        │  - Pricing + Surge          │
        │  - Demand (Poisson)         │
        │  - Events                   │
        │  - Fraud engine             │
        └─────────────────────────────┘
```

---

## Running real LLMs

### `claude_baseline` — direct Anthropic SDK

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python eval/run.py --tasks "rideshare/*" --agent claude_baseline --model claude-opus-4-7 --n-episodes 3
```

### `litellm` — any OpenAI-compatible endpoint (OpenAI, NCSU LAS LiteLLM, Together, Groq, local vLLM)

```bash
# Set OPENAI_API_KEY or LAS_API_TOKEN in your shell first.
# Default base_url is the LAS LiteLLM proxy; override with --base-url.
python eval/run.py --tasks "rideshare/*" --agent litellm --model openai/gpt-4o --n-episodes 3

# See what models a given endpoint exposes:
python eval/list_models.py
python eval/list_models.py --filter gpt
python eval/list_models.py --filter claude
```

The `litellm` agent speaks the OpenAI Chat Completions tool-calling protocol so a single agent can drive GPT-4o, GPT-5, Claude (via Bedrock), Llama 3.3, Gemini, and dozens of others without code changes.

---

## Sample scorecards

### Gold oracle, clean (24 episodes)

```
=== Scorecard (24 episodes) ===
Overall success: 83.3%  mean reward: 0.92
  rideshare/match_single_ride                100%
  rideshare/refund_cancelled_trip            100%
  rideshare/verify_driver_documents          100%
  rideshare/fraud_ring_detection             100%
  rideshare/lost_item_recovery               100%
  rideshare/driver_pay_dispute               100%
  rideshare/accident_incident_response       100%
  rideshare/account_takeover_response        100%
  rideshare/event_surge_planning             100%
  rideshare/coordinated_fraud_response       100%
  rideshare/realtime_dispatch_window          0%   reward 0.50  ← KPI ceiling
  rideshare/surge_demand_spike                0%   reward 0.50  ← unmatched ratio
```

The 2 fail-with-half-credit tasks are intentional benchmark headroom — even the optimal greedy oracle can't fully clear the KPI bar on real-time multi-objective dispatching. That gap is what stronger agents are scored against.

---

## How this could be used to actually train a model

The trajectory format is directly consumable by training pipelines:

1. **Supervised fine-tuning (SFT).** Take all trajectories where `success == True`. For each step, the prompt is the message history; the target is the action that was taken. Fine-tune a smaller model (Llama 3.3 8B, Haiku) to imitate. Cheap path to a specialist.
2. **Direct preference optimisation (DPO).** Pair a successful trajectory with a failed one on the same task + seed. The model learns to prefer the successful action sequence.
3. **Reward modelling + PPO.** Train a separate model to predict the verifier's reward from observation + action. Use it in a PPO loop to update the agent's policy. The simulator's tick-by-tick determinism (under fixed seed) makes this tractable.
4. **Inference-time scaling.** Run N rollouts per task, take the best one by reward.

The gym does double duty: **benchmark** (rank agents) AND **data factory** (produce training data for the next agent).

---

## What's not built (post-MVP)

- Toxiproxy network-level chaos in the eval harness (Docker compose is wired but middleware-level perturbations cover the same failure modes in-process).
- More tasks. The plan calls for ~30 tasks total via combinatorial perturbation expansion of the 12 base tasks (mechanical to add).
- Real OSM city graph (osmnx). The 6-zone stylized city is sufficient for these tasks; OSM would add ~200MB of dep + city cache and is a separate project.
- WebSocket streaming of world events to the agent (currently the agent polls via list_pending_requests / tick).
- A dedicated agent persona for "driver-side" or "rider-side" sub-tasks (currently we have a unified "platform brain" persona).

---

## Sister project

[github.com/dhirengshetty14/shopify-gym](https://github.com/dhirengshetty14/shopify-gym) — same architecture, scoped to e-commerce ops (Shopify-style chargebacks, fraud rings, oversells, abandoned carts). 4 tasks instead of 12, no real-time simulator. Useful for understanding the shared scaffolding patterns.

---

## License

MIT.
