"""Task registry for the 12 rideshare-gym tasks."""

from __future__ import annotations

from collections.abc import Callable

from rideshare_gym.core.task import AbstractTask
from rideshare_gym.tasks._base import RideshareTask  # noqa: F401

# Easy
from rideshare_gym.tasks.match_single_ride import MatchSingleRideTask
from rideshare_gym.tasks.refund_cancelled_trip import RefundCancelledTripTask
from rideshare_gym.tasks.verify_driver_documents import VerifyDriverDocumentsTask

# Medium
from rideshare_gym.tasks.surge_demand_spike import SurgeDemandSpikeTask
from rideshare_gym.tasks.fraud_ring_detection import FraudRingDetectionTask
from rideshare_gym.tasks.lost_item_recovery import LostItemRecoveryTask
from rideshare_gym.tasks.driver_pay_dispute import DriverPayDisputeTask
from rideshare_gym.tasks.accident_incident_response import AccidentIncidentResponseTask
from rideshare_gym.tasks.account_takeover_response import AccountTakeoverResponseTask

# Hard
from rideshare_gym.tasks.realtime_dispatch_window import RealtimeDispatchWindowTask
from rideshare_gym.tasks.event_surge_planning import EventSurgePlanningTask
from rideshare_gym.tasks.coordinated_fraud_response import CoordinatedFraudResponseTask


REGISTRY: dict[str, Callable[..., AbstractTask]] = {
    cls.task_id: cls for cls in [
        MatchSingleRideTask,
        RefundCancelledTripTask,
        VerifyDriverDocumentsTask,
        SurgeDemandSpikeTask,
        FraudRingDetectionTask,
        LostItemRecoveryTask,
        DriverPayDisputeTask,
        AccidentIncidentResponseTask,
        AccountTakeoverResponseTask,
        RealtimeDispatchWindowTask,
        EventSurgePlanningTask,
        CoordinatedFraudResponseTask,
    ]
}


def make(task_id: str, **kwargs) -> AbstractTask:
    if task_id not in REGISTRY:
        raise KeyError(f"unknown task: {task_id}; known: {sorted(REGISTRY)}")
    return REGISTRY[task_id](**kwargs)


def all_task_ids() -> list[str]:
    return sorted(REGISTRY)


__all__ = [
    "REGISTRY",
    "RideshareTask",
    "make",
    "all_task_ids",
    "MatchSingleRideTask", "RefundCancelledTripTask", "VerifyDriverDocumentsTask",
    "SurgeDemandSpikeTask", "FraudRingDetectionTask", "LostItemRecoveryTask",
    "DriverPayDisputeTask", "AccidentIncidentResponseTask",
    "AccountTakeoverResponseTask",
    "RealtimeDispatchWindowTask", "EventSurgePlanningTask",
    "CoordinatedFraudResponseTask",
]
