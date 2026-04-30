"""Gold oracle agent — knows ground truth, solves all 12 rideshare tasks deterministically.

Used as the upper-bound baseline. Real-time tasks (H1) use a nearest-driver greedy
policy + adaptive surge; ops tasks read ground truth straight from sandbox metadata.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.core.recorder import Trajectory, TrajectoryRecorder
from rideshare_gym.core.types import ToolCall


class GoldOracleAgent:
    """Solves the 12 MVP tasks deterministically. Useful as upper-bound baseline."""

    def run(
        self,
        env: GymEnvironment,
        *,
        on_step=None,
        on_event=None,
    ) -> Trajectory:
        self._on_step = on_step
        self._on_event = on_event
        obs, info = env.reset()
        if on_event:
            on_event({"event": "episode_start", "task_id": info["task_id"]})

        rec = TrajectoryRecorder(
            task_id=info["task_id"],
            seed=env.task.seed,
            ground_truth=env.ground_truth,
            perturbations=info["perturbations"],
            meta={"agent_id": "gold_oracle"},
        )
        rec.set_initial(info["initial_state_hash"])
        gt: dict[str, Any] = env.sandbox.rs.get_metadata().get("ground_truth", {})

        base = info["task_id"].split("__")[0]
        success = False
        try:
            if base == "rideshare/match_single_ride":
                success = self._solve_match(env, rec, gt)
            elif base == "rideshare/refund_cancelled_trip":
                success = self._solve_refund_cancelled(env, rec, gt)
            elif base == "rideshare/verify_driver_documents":
                success = self._solve_verify_docs(env, rec, gt)
            elif base == "rideshare/surge_demand_spike":
                success = self._solve_surge_spike(env, rec, gt)
            elif base == "rideshare/fraud_ring_detection":
                success = self._solve_fraud_ring(env, rec, gt)
            elif base == "rideshare/lost_item_recovery":
                success = self._solve_lost_item(env, rec, obs.data, gt)
            elif base == "rideshare/driver_pay_dispute":
                success = self._solve_pay_dispute(env, rec, gt)
            elif base == "rideshare/accident_incident_response":
                success = self._solve_incident(env, rec, gt)
            elif base == "rideshare/account_takeover_response":
                success = self._solve_takeover(env, rec, gt)
            elif base == "rideshare/realtime_dispatch_window":
                success = self._solve_realtime(env, rec, gt)
            elif base == "rideshare/event_surge_planning":
                success = self._solve_event_surge(env, rec, gt)
            elif base == "rideshare/coordinated_fraud_response":
                success = self._solve_coordinated_fraud(env, rec, gt)
        except Exception:
            success = False

        traj = rec.finalize(
            final_state_hash=env.final_state_hash,
            success=success,
            error_category=None if success else "goal_incomplete",
        )
        if on_event:
            on_event({"event": "finished", "success": success,
                      "final_reward": traj.final_reward,
                      "error_category": traj.error_category})
        return traj

    # ----- helpers ----- #
    def _step(self, env, rec, call: ToolCall):
        obs, reward, terminated, truncated, info = env.step(call)
        step_idx = len(rec.trajectory.steps)
        rec.record(call, obs, reward, terminated, truncated,
                   latency_ms=info.get("tool_latency_ms", 0.0), info=info)
        if getattr(self, "_on_step", None):
            self._on_step(step_idx, call, obs, reward, terminated, info)
        return obs, reward, terminated, info

    # ===== solvers ===== #

    def _solve_match(self, env, rec, gt) -> bool:
        # Use auto_match_nearest with the trip id from ground truth.
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="auto_match_nearest",
            arguments={"trip_id": gt["trip_id"]},
        ))
        return term

    def _solve_refund_cancelled(self, env, rec, gt) -> bool:
        amount = gt["expected_refund_amount"]
        self._step(env, rec, ToolCall(
            name="create_refund",
            arguments={"trip_id": gt["trip_id"], "amount": amount,
                       "reason": "driver_cancelled", "notify_rider": True},
        ))
        # Even though create_refund auto-notifies, explicitly notify too — tasks
        # that count "rider_notified" pass either way.
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="send_to_rider",
            arguments={"rider_id": gt["rider_id"], "template": "trip.refunded",
                       "variables": {"trip_id": gt["trip_id"]}},
        ))
        return term

    def _solve_verify_docs(self, env, rec, gt) -> bool:
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="verify_driver_documents",
            arguments={"driver_id": gt["driver_id"]},
        ))
        return term

    def _solve_surge_spike(self, env, rec, gt) -> bool:
        zone = gt["affected_zone"]
        # Raise surge.
        self._step(env, rec, ToolCall(
            name="set_surge",
            arguments={"zone_id": zone, "multiplier": 2.0, "ttl_minutes": 5.0},
        ))
        # Match as many pending as possible across 6 ticks.
        for _ in range(6):
            obs, _, _, _ = self._step(env, rec, ToolCall(
                name="list_pending_requests",
                arguments={"zone_id": zone, "limit": 50},
            ))
            pending = obs.data.get("pending", [])[:8]
            for req in pending:
                self._step(env, rec, ToolCall(
                    name="auto_match_nearest",
                    arguments={"trip_id": req["trip_id"]},
                ))
            obs, _, term, _ = self._step(env, rec, ToolCall(
                name="tick", arguments={"dt_seconds": 60.0},
            ))
            if term:
                return True
        return term  # may be False; verifier evaluates KPIs at the last step

    def _solve_fraud_ring(self, env, rec, gt) -> bool:
        ring_ids = gt["fraud_rider_ids"]
        for rid in ring_ids:
            _, _, term, _ = self._step(env, rec, ToolCall(
                name="freeze_account",
                arguments={"target_kind": "rider", "target_id": rid,
                           "reason": "fraud_ring"},
            ))
        # bonus: ban shared device fingerprint
        self._step(env, rec, ToolCall(
            name="ban_device",
            arguments={"fingerprint": gt["fingerprint"], "reason": "fraud_ring"},
        ))
        return term

    def _solve_lost_item(self, env, rec, init_state, gt) -> bool:
        out, _, _, _ = self._step(env, rec, ToolCall(
            name="create_lost_item",
            arguments={"trip_id": gt["trip_id"], "description": "wallet"},
        ))
        li_id = out.data["lost_item"]["id"]
        self._step(env, rec, ToolCall(
            name="assign_lost_item",
            arguments={"lost_item_id": li_id,
                       "driver_id": gt["driver_id"],
                       "return_method": "next_idle_window"},
        ))
        # schedule return-pickup notifies both parties with the code.
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="schedule_lost_item_pickup",
            arguments={"lost_item_id": li_id,
                       "pickup_at": 600.0,
                       "pickup_location": [15.0, 15.0],
                       "notify_rider": True, "notify_driver": True},
        ))
        return term

    def _solve_pay_dispute(self, env, rec, gt) -> bool:
        amt = gt["underpayment_dollars"]
        self._step(env, rec, ToolCall(
            name="adjust_driver_payout",
            arguments={"driver_id": gt["driver_id"], "amount": amt,
                       "reason": "surge_correction",
                       "related_trip_id": gt["trip_id"]},
        ))
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="send_to_driver",
            arguments={"driver_id": gt["driver_id"],
                       "template": "payout.adjusted",
                       "variables": {"trip_id": gt["trip_id"], "amount": amt}},
        ))
        return term

    def _solve_incident(self, env, rec, gt) -> bool:
        self._step(env, rec, ToolCall(
            name="escalate_incident",
            arguments={"incident_id": gt["incident_id"], "level": 2,
                       "notify_parties": True},
        ))
        self._step(env, rec, ToolCall(
            name="contact_emergency",
            arguments={"incident_id": gt["incident_id"], "kind": "911"},
        ))
        self._step(env, rec, ToolCall(
            name="create_refund",
            arguments={"trip_id": gt["trip_id"],
                       "amount": gt["expected_refund_amount"],
                       "reason": "safety_incident", "notify_rider": True},
        ))
        self._step(env, rec, ToolCall(
            name="adjust_driver_payout",
            arguments={"driver_id": gt["driver_id"], "amount": 25.0,
                       "reason": "incident_compensation",
                       "related_trip_id": gt["trip_id"]},
        ))
        # Explicit messages just to be safe (refund/escalate already notify).
        self._step(env, rec, ToolCall(
            name="send_to_rider",
            arguments={"rider_id": gt["rider_id"],
                       "template": "safety.incident_resolved",
                       "variables": {"incident_id": gt["incident_id"]}},
        ))
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="send_to_driver",
            arguments={"driver_id": gt["driver_id"],
                       "template": "safety.incident_resolved",
                       "variables": {"incident_id": gt["incident_id"]}},
        ))
        return term

    def _solve_takeover(self, env, rec, gt) -> bool:
        self._step(env, rec, ToolCall(
            name="freeze_rider",
            arguments={"rider_id": gt["rider_id"],
                       "reason": "anomalous_login_pattern"},
        ))
        self._step(env, rec, ToolCall(
            name="login_history",
            arguments={"rider_id": gt["rider_id"]},
        ))
        self._step(env, rec, ToolCall(
            name="send_to_rider",
            arguments={"rider_id": gt["rider_id"],
                       "template": "account.takeover_check",
                       "variables": {"reason": "verify_identity"}},
        ))
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="restore_rider",
            arguments={"rider_id": gt["rider_id"], "reason": "verified"},
        ))
        return term

    def _solve_realtime(self, env, rec, gt) -> bool:
        # 60 ticks of 30s each. Each tick: drain the pending queue (auto-match),
        # then advance time.
        max_steps = env.task.max_steps - 5
        steps_done = 0
        max_sim = gt["max_sim_seconds"]
        for _ in range(70):
            if steps_done > max_steps:
                break
            obs, _, _, info = self._step(env, rec, ToolCall(
                name="list_pending_requests",
                arguments={"limit": 12},
            ))
            steps_done += 1
            pending = obs.data.get("pending", [])
            for req in pending[:8]:
                if steps_done > max_steps:
                    break
                self._step(env, rec, ToolCall(
                    name="auto_match_nearest",
                    arguments={"trip_id": req["trip_id"]},
                ))
                steps_done += 1
            obs, _, term, info = self._step(env, rec, ToolCall(
                name="tick", arguments={"dt_seconds": 30.0},
            ))
            steps_done += 1
            if term:
                return True
            if info.get("verifier", {}).get("info", {}).get("now", 0) >= max_sim:
                break
            now = obs.data.get("now", 0)
            if now is not None and now >= max_sim:
                break
        # Final assessment via the verifier (it's checked every step automatically).
        return term

    def _solve_event_surge(self, env, rec, gt) -> bool:
        # Rebalance 8 drivers to stadium (more than min 6 to be safe).
        obs, _, _, _ = self._step(env, rec, ToolCall(
            name="list_idle_drivers",
            arguments={"limit": 50},
        ))
        idle = obs.data.get("idle_drivers", [])
        moved = 0
        for d in idle:
            if d.get("home_zone_id") == "stadium":
                continue
            self._step(env, rec, ToolCall(
                name="rebalance_driver",
                arguments={"driver_id": d["driver_id"],
                           "target_zone_id": "stadium"},
            ))
            moved += 1
            if moved >= 8:
                break
        # Set surge.
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="set_surge",
            arguments={"zone_id": "stadium", "multiplier": 2.0,
                       "ttl_minutes": 10.0},
        ))
        return term

    def _solve_coordinated_fraud(self, env, rec, gt) -> bool:
        for rid in gt["ring_rider_ids"]:
            self._step(env, rec, ToolCall(
                name="freeze_account",
                arguments={"target_kind": "rider", "target_id": rid,
                           "reason": "fraud_ring"},
            ))
        self._step(env, rec, ToolCall(
            name="ban_device",
            arguments={"fingerprint": gt["fingerprint"], "reason": "fraud_ring"},
        ))
        for did, amt_str in gt["victim_amounts"].items():
            self._step(env, rec, ToolCall(
                name="adjust_driver_payout",
                arguments={"driver_id": int(did), "amount": float(amt_str),
                           "reason": "fraud_ring_compensation"},
            ))
        # Final no-op call so the verifier runs after the last adjustment.
        _, _, term, _ = self._step(env, rec, ToolCall(
            name="list_flagged_accounts", arguments={},
        ))
        return term
