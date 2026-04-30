"""H3 — Coordinated fraud response under time pressure.

Setup: 5-account fraud ring + 4 victim drivers (overcharged via chargebacks).
Agent must: freeze all 5 ring accounts, ban shared device fingerprint,
adjust the 4 victim drivers' payouts upward, message all 5 affected legit
riders whose accounts may have been spoofed.

Verifier (composite all_of):
- All 5 ring riders frozen
- Shared device fingerprint banned
- All 4 victim drivers got positive payout adjustments
- No legit riders frozen (FP=0)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rideshare_gym.core.task import InitialState
from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.verifier import (
    AssertionListVerifier, CompositeVerifier, Verifier,
)
from rideshare_gym.mock_server.seed import seed_drivers, seed_riders
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.rideshare_sandbox import RideshareSandbox
from rideshare_gym.tasks._base import RideshareTask
from rideshare_gym.tools import select
from rideshare_gym.world.fraud_engine import plant_fraud_ring


class CoordinatedFraudResponseTask(RideshareTask):
    task_id = "rideshare/coordinated_fraud_response"
    difficulty = "hard"
    max_steps = 60

    def tools(self) -> list[ToolSpec]:
        return select(
            "list_flagged_accounts", "list_flagged_trips",
            "cluster_by_fingerprint", "freeze_account", "ban_device",
            "adjust_driver_payout", "send_to_rider",
            "get_rider", "list_trips",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        # Baseline drivers + riders.
        driver_ids = seed_drivers(w, n_drivers=12, seed=self.seed)
        seed_riders(w, n_riders=20, seed=self.seed)

        # Plant the fraud ring.
        ring = plant_fraud_ring(
            w, kind="account_farm", n_riders=5, seed=self.seed,
            obvious_count=4,
        )

        # Pick 4 driver victims (overcharged via chargebacks).
        victim_drivers = driver_ids[:4]
        victim_owed_by_driver = {
            did: Decimal(str(round(35.0 + (i * 8.5), 2)))
            for i, did in enumerate(victim_drivers)
        }
        ground_truth = {
            "ring_rider_ids": ring.rider_ids,
            "fingerprint": ring.fingerprint,
            "victim_drivers": list(victim_drivers),
            "victim_amounts": {str(k): str(v)
                                for k, v in victim_owed_by_driver.items()},
        }
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                "COORDINATED ATTACK: 5-account fraud ring is in the middle of "
                "an attack. Required actions: "
                "(1) freeze ALL 5 ring riders, "
                "(2) ban the shared device fingerprint, "
                "(3) adjust payout upward for each of the 4 victim drivers "
                "(amounts in metadata.ground_truth.victim_amounts), "
                "(4) do NOT freeze any legit accounts."
            ),
            snapshot={
                "n_ring": len(ring.rider_ids),
                "n_victim_drivers": len(victim_drivers),
                "fingerprint": ring.fingerprint[:8] + "...",
            },
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            gt = w.metadata.get("ground_truth", {})
            ring_target = set(gt.get("ring_rider_ids", []))
            fp = gt.get("fingerprint", "")
            victim_drivers = set(gt.get("victim_drivers", []))
            victim_amounts = {int(k): Decimal(v)
                              for k, v in (gt.get("victim_amounts") or {}).items()}

            frozen_ring = {r.id for r in w.riders.values() if "frozen" in r.flags}
            non_ring_frozen = frozen_ring - ring_target
            adjusted_drivers = {a.driver_id for a in w.payout_adjustments}
            sufficient_adjustments = 0
            for did in victim_drivers:
                received = sum(
                    (a.amount for a in w.payout_adjustments if a.driver_id == did),
                    start=Decimal("0"),
                )
                expected = victim_amounts.get(did, Decimal("0"))
                if abs(received - expected) <= Decimal("0.50"):
                    sufficient_adjustments += 1

            return {
                "ring_freeze_complete":
                    ring_target.issubset(frozen_ring),
                "no_legit_frozen": len(non_ring_frozen) == 0,
                "device_banned": fp in w.banned_devices,
                "victim_drivers_compensated": sufficient_adjustments,
                "n_expected_victims": len(victim_drivers),
            }

        return CompositeVerifier(
            children=[
                AssertionListVerifier(
                    assertions=[
                        ("ring_freeze_complete",
                            lambda s: s["ring_freeze_complete"]),
                        ("no_legit_riders_frozen",
                            lambda s: s["no_legit_frozen"]),
                        ("device_fingerprint_banned",
                            lambda s: s["device_banned"]),
                        ("all_victim_drivers_compensated",
                            lambda s: s["victim_drivers_compensated"]
                                       == s["n_expected_victims"]),
                    ],
                    snapshot_fn=_snap,
                ),
            ],
            mode="all_of",
        )
