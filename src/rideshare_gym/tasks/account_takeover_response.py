"""M6 — Account takeover response.

Setup: A rider account is suddenly taking trips at unusual hour, in unusual
city, with a new device fingerprint. Agent must freeze, review, message via
verified channel, and after the (simulated) verify flag is set, restore.
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
from rideshare_gym.world.riders import Rider


class AccountTakeoverResponseTask(RideshareTask):
    task_id = "rideshare/account_takeover_response"
    difficulty = "medium"
    max_steps = 16

    def tools(self) -> list[ToolSpec]:
        return select(
            "get_rider", "login_history",
            "freeze_rider", "restore_rider", "send_to_rider",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        rid = w.next_id()
        # Rider has been around 2 years, normal pattern.
        w.riders[rid] = Rider(
            id=rid, name="targeted rider",
            email="targeted@example.com",
            payment_method_id=f"pm_{rid}",
            home_zone_id="suburb_n",
            typical_login_zone_id="suburb_n",
            typical_login_hour_window=(7, 22),
            created_at="2024-04-01",
            device_fingerprint=f"old_dfp_{rid}",
        )
        # Plant some "anomalies" for the login_history endpoint to surface.
        w.metadata.setdefault("login_anomalies", {})[str(rid)] = [
            {"at_hour": 3, "zone_id": "downtown",
             "device_fingerprint": f"NEW_dfp_{rid}",
             "ip_country": "RU", "note": "unusual hour + new device + foreign IP"},
            {"at_hour": 4, "zone_id": "downtown",
             "device_fingerprint": f"NEW_dfp_{rid}",
             "ip_country": "RU"},
        ]
        # Also set verified_owner=True in metadata; the agent reaches a state
        # where after confirming with the real owner (via send_to_rider), they
        # call restore_rider. The test of correctness is the sequence of
        # actions, not whether identity was actually checked.
        ground_truth = {"rider_id": rid}
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                f"FRAUD ALERT: Rider {rid}'s account has anomalous logins "
                f"(3am, new device, foreign IP). Their normal pattern is "
                f"daytime use from suburb_n on a stable device. Required "
                f"steps: 1) freeze the account immediately; 2) review login "
                f"history; 3) message the real owner via verified channel "
                f"(template account.takeover_check); 4) after positive "
                f"verification, restore the account."
            ),
            snapshot={"rider_id": rid},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            gt = w.metadata.get("ground_truth", {})
            rid = gt.get("rider_id", -1)
            r = w.riders.get(rid)
            actions = [
                a for a in w.account_actions
                if a.target_kind == "rider" and a.target_id == str(rid)
            ]
            kinds = [a.action for a in actions]
            messages = [
                m for m in w.sent_messages if m.to == f"rider:{rid}"
            ]
            return {
                "ever_frozen": "freeze" in kinds,
                "ever_restored": "restore" in kinds,
                "current_frozen": bool(r) and "frozen" in r.flags,
                "n_messages": len(messages),
            }

        return AssertionListVerifier(
            assertions=[
                ("account_frozen_at_some_point",
                    lambda s: s["ever_frozen"]),
                ("account_restored_after_verification",
                    lambda s: s["ever_restored"]),
                ("currently_restored",
                    lambda s: not s["current_frozen"]),
                ("owner_messaged_via_verified_channel",
                    lambda s: s["n_messages"] >= 1),
            ],
            snapshot_fn=_snap,
        )
