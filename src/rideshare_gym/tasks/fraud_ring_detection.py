"""M2 — Detect a 5-account fraud ring among recent trips.

Setup: 50 recent trips; a 5-account ring shares device fingerprint + payment
BIN. 2 of 5 are subtle (no obvious risk_level=high).
Verifier: F1 >= 0.9 against ground-truth ring; FPs penalised 2x FN.
"""

from __future__ import annotations

from typing import Any

from rideshare_gym.core.task import InitialState
from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.verifier import MetricThresholdVerifier, Verifier
from rideshare_gym.mock_server.seed import seed_drivers, seed_riders
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.rideshare_sandbox import RideshareSandbox
from rideshare_gym.tasks._base import RideshareTask
from rideshare_gym.tools import select
from rideshare_gym.world.fraud_engine import plant_fraud_ring


class FraudRingDetectionTask(RideshareTask):
    task_id = "rideshare/fraud_ring_detection"
    difficulty = "hard"
    max_steps = 50

    def tools(self) -> list[ToolSpec]:
        return select(
            "list_trips", "get_trip", "get_rider",
            "list_flagged_accounts", "list_flagged_trips",
            "cluster_by_fingerprint", "freeze_account", "ban_device",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        # Plant baseline.
        seed_drivers(w, n_drivers=15, seed=self.seed)
        seed_riders(w, n_riders=45, seed=self.seed)

        ring = plant_fraud_ring(
            w, kind="account_farm", n_riders=5, seed=self.seed,
            obvious_count=3,
        )
        ground_truth = {
            "fraud_rider_ids": ring.rider_ids,
            "fingerprint": ring.fingerprint,
            "bin": ring.bin,
        }
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                "FRAUD ALERT: Recent chargebacks suggest an account-farm "
                "ring. 5 rider accounts share a device fingerprint and "
                "payment-card BIN. 3 are flagged high-risk by automatic "
                "scoring; 2 are subtle. Find ALL 5 and freeze each "
                "(reason='fraud_ring'). Bonus: ban the shared device "
                "fingerprint. Do NOT freeze legit riders."
            ),
            snapshot={
                "n_rider_accounts": len(w.riders),
                "hint": ("Inspect flagged accounts first, then look at all "
                          "riders sharing the same device_fingerprint or BIN."),
            },
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _f1_with_fp_penalty(sandbox: RideshareSandbox, _gt: dict) -> float:
            w = get_world(sandbox.tenant_id)
            gt = w.metadata.get("ground_truth", {})
            target = set(gt.get("fraud_rider_ids", []))
            if not target:
                return 0.0
            frozen_riders = {
                r.id for r in w.riders.values() if "frozen" in r.flags
            }
            tp = len(frozen_riders & target)
            fp = len(frozen_riders - target)
            fn = len(target - frozen_riders)
            if tp == 0 and fp == 0:
                return 0.0
            precision = tp / (tp + 2 * fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        return MetricThresholdVerifier(
            metric_fn=_f1_with_fp_penalty,
            ground_truth={},
            threshold=0.9,
            metric_name="fp_weighted_f1",
        )
