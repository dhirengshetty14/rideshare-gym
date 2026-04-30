"""E3 — Verify a driver's expired documents.

Setup: 1 driver with expired drivers_license.
Verifier: docs_verified flips to True after agent calls verify_driver_documents.
"""

from __future__ import annotations

from typing import Any

from rideshare_gym.core.task import InitialState
from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.verifier import AssertionListVerifier, Verifier
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.rideshare_sandbox import RideshareSandbox
from rideshare_gym.tasks._base import RideshareTask
from rideshare_gym.tools import select
from rideshare_gym.world.drivers import Driver, DriverStatus


class VerifyDriverDocumentsTask(RideshareTask):
    task_id = "rideshare/verify_driver_documents"
    difficulty = "easy"
    max_steps = 8

    def tools(self) -> list[ToolSpec]:
        return select(
            "get_driver", "get_driver_documents",
            "verify_driver_documents", "send_to_driver",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        did = w.next_id()
        w.drivers[did] = Driver(
            id=did, name="driver with expired docs",
            location=(15.0, 15.0),
            status=DriverStatus.OFFLINE,
            vehicle_type="uberx",
            docs_verified=False,
            docs_expiry="2026-04-25",       # 5 days ago in our episode timeline
            home_zone_id="downtown",
        )
        ground_truth = {"driver_id": did}
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                f"Driver {did}'s documents (drivers_license) expired 5 days "
                f"ago and they're blocked from accepting trips. Verify "
                f"their documents to re-enable them."
            ),
            snapshot={"driver_id": did, "docs_verified": False},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            gt = w.metadata.get("ground_truth", {})
            d = w.drivers.get(gt.get("driver_id", -1))
            return {
                "docs_verified": d.docs_verified if d else False,
                "driver_id": d.id if d else None,
            }

        return AssertionListVerifier(
            assertions=[
                ("docs_verified", lambda s: s["docs_verified"] is True),
            ],
            snapshot_fn=_snap,
        )
