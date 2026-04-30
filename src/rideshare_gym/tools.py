"""Rideshare-specific ToolSpec definitions (~40 tools).

Tasks pick subsets via `select(...)`. Selection drives action-space size,
which drives task difficulty.
"""

from __future__ import annotations

from typing import Any

from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.types import ToolResult
from rideshare_gym.rideshare_sandbox import RideshareSandbox


def _ok(payload: dict[str, Any], summary: str = "") -> ToolResult:
    return ToolResult(ok=True, payload=payload, summary=summary or "ok")


# ----- marketplace ----- #

LIST_PENDING_REQUESTS = ToolSpec(
    name="list_pending_requests",
    description="List pending ride requests (status=requested), newest-first. Filter by zone_id.",
    input_schema={
        "type": "object",
        "properties": {
            "zone_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.list_pending_requests(**a), "pending"),
)

LIST_IDLE_DRIVERS = ToolSpec(
    name="list_idle_drivers",
    description="List idle drivers available for dispatch. Filter by zone_id and/or vehicle_type.",
    input_schema={
        "type": "object",
        "properties": {
            "zone_id": {"type": "string"},
            "vehicle_type": {"type": "string", "enum": ["uberx", "uberxl", "uberblack"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.list_idle_drivers(**a), "idle drivers"),
)

MATCH_RIDE = ToolSpec(
    name="match_ride",
    description="Dispatch the given driver to the given pending trip. Driver must be idle, with verified docs and not frozen.",
    input_schema={
        "type": "object", "required": ["trip_id", "driver_id"],
        "properties": {
            "trip_id": {"type": "integer"},
            "driver_id": {"type": "integer"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.match_ride(**a), f"matched {a['trip_id']}->{a['driver_id']}"),
)

CANCEL_TRIP = ToolSpec(
    name="cancel_trip",
    description="Admin-cancel a trip with a reason. cancelled_by is one of rider/driver/system.",
    input_schema={
        "type": "object", "required": ["trip_id", "reason"],
        "properties": {
            "trip_id": {"type": "integer"},
            "reason": {"type": "string", "enum": [
                "customer_no_show", "customer_changed_mind", "driver_too_far",
                "driver_no_show", "vehicle_issue", "safety_concern",
                "fraud_suspected", "other",
            ]},
            "cancelled_by": {"type": "string",
                              "enum": ["rider", "driver", "system"], "default": "system"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.cancel_trip_admin(**a), f"cancelled {a['trip_id']}"),
)

AUTO_MATCH_NEAREST = ToolSpec(
    name="auto_match_nearest",
    description="Convenience: match the given pending trip to its nearest idle driver by ETA. Returns 409 if no driver available.",
    input_schema={
        "type": "object", "required": ["trip_id"],
        "properties": {"trip_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.auto_match_nearest(a["trip_id"]),
                              f"auto-matched {a['trip_id']}"),
)

SET_SURGE = ToolSpec(
    name="set_surge",
    description="Override surge multiplier on a zone. Multiplier in [1.0, 5.0]. ttl_minutes controls how long override holds.",
    input_schema={
        "type": "object", "required": ["zone_id", "multiplier"],
        "properties": {
            "zone_id": {"type": "string"},
            "multiplier": {"type": "number", "minimum": 1.0, "maximum": 5.0},
            "ttl_minutes": {"type": "number", "default": 5.0},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.set_surge(**a), f"surge {a['zone_id']}={a['multiplier']}"),
)

REBALANCE_DRIVER = ToolSpec(
    name="rebalance_driver",
    description="Move an idle driver to a different zone. Driver must currently be idle.",
    input_schema={
        "type": "object", "required": ["driver_id", "target_zone_id"],
        "properties": {
            "driver_id": {"type": "integer"},
            "target_zone_id": {"type": "string"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.rebalance_driver(**a),
                              f"rebalanced {a['driver_id']}"),
)

OFFER_INCENTIVE = ToolSpec(
    name="offer_incentive",
    description="Offer a bonus to an idle driver to encourage them to head to a hot zone or take a shift.",
    input_schema={
        "type": "object", "required": ["driver_id"],
        "properties": {
            "driver_id": {"type": "integer"},
            "type": {"type": "string", "default": "bonus"},
            "amount": {"type": "number", "default": 10.0},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.offer_incentive(**a),
                              f"incentive {a['driver_id']}"),
)

ZONE_SUPPLY_DEMAND = ToolSpec(
    name="zone_supply_demand",
    description="Per-zone snapshot of pending demand, idle supply, and current surge multiplier.",
    input_schema={"type": "object", "properties": {}},
    handler=lambda sb, a: _ok(sb.rs.zone_supply_demand(), "supply/demand"),
)

DISPATCH_LOG = ToolSpec(
    name="dispatch_log",
    description="Recent matching decisions made by anyone (this agent or earlier). Each entry includes trip_id, driver_id, decided_at, eta_minutes, alternatives_considered.",
    input_schema={"type": "object", "properties": {
        "limit": {"type": "integer", "default": 100, "maximum": 2000}}},
    handler=lambda sb, a: _ok(sb.rs.dispatch_log(**a), "dispatch log"),
)

TICK = ToolSpec(
    name="tick",
    description="Advance simulator time by `dt_seconds`. Real-time tasks call this to let drivers complete trips and new demand to arrive. Default 30s.",
    input_schema={
        "type": "object",
        "properties": {
            "dt_seconds": {"type": "number", "minimum": 1.0, "maximum": 600.0,
                            "default": 30.0},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.tick(a.get("dt_seconds", 30.0)),
                              f"+{a.get('dt_seconds', 30.0)}s"),
)


# ----- trips ----- #

LIST_TRIPS = ToolSpec(
    name="list_trips",
    description="List trips with optional filters: rider_id, driver_id, status (requested/matched/in_trip/completed/cancelled).",
    input_schema={
        "type": "object",
        "properties": {
            "rider_id": {"type": "integer"},
            "driver_id": {"type": "integer"},
            "status": {"type": "string"},
            "limit": {"type": "integer", "default": 100, "maximum": 500},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.list_trips(**a), "trips"),
)

GET_TRIP = ToolSpec(
    name="get_trip",
    description="Fetch a single trip. Includes pickup/dropoff zones, fare, ratings, side records (refund/dispute/lost_item/incident), rider payment_bin and device_fingerprint.",
    input_schema={
        "type": "object", "required": ["trip_id"],
        "properties": {"trip_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.get_trip(a["trip_id"]), f"trip {a['trip_id']}"),
)

GET_TRIP_GPS_LOG = ToolSpec(
    name="get_trip_gps_log",
    description="Fetch the per-tick GPS waypoints for a trip. Useful for forensic pay/distance verification and gps-spoofing detection.",
    input_schema={
        "type": "object", "required": ["trip_id"],
        "properties": {"trip_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.get_trip_gps_log(a["trip_id"]),
                              f"gps {a['trip_id']}"),
)


# ----- refunds + disputes ----- #

CREATE_REFUND = ToolSpec(
    name="create_refund",
    description="Issue a refund on a trip. Amount is in dollars. Reason is human-readable. Optionally notifies rider via push.",
    input_schema={
        "type": "object", "required": ["trip_id", "amount", "reason"],
        "properties": {
            "trip_id": {"type": "integer"},
            "amount": {"type": "number", "minimum": 0},
            "reason": {"type": "string"},
            "notify_rider": {"type": "boolean", "default": True},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.create_refund(**a), f"refund {a['trip_id']}"),
)

LIST_REFUNDS = ToolSpec(
    name="list_refunds",
    description="List refunds. Optionally filter by trip_id.",
    input_schema={"type": "object", "properties": {"trip_id": {"type": "integer"}}},
    handler=lambda sb, a: _ok(sb.rs.list_refunds(**a), "refunds"),
)

LIST_DISPUTES = ToolSpec(
    name="list_disputes",
    description="List chargeback disputes, optionally filtered by status (needs_response/under_review/won/lost).",
    input_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    handler=lambda sb, a: _ok(sb.rs.list_disputes(**a), "disputes"),
)

GET_DISPUTE = ToolSpec(
    name="get_dispute",
    description="Fetch a single dispute including reason, deadline, and submitted response (if any).",
    input_schema={
        "type": "object", "required": ["dispute_id"],
        "properties": {"dispute_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.get_dispute(a["dispute_id"]),
                              f"dispute {a['dispute_id']}"),
)

SUBMIT_DISPUTE_RESPONSE = ToolSpec(
    name="submit_dispute_response",
    description="Submit chargeback dispute response. Common evidence fields: tracking_log_url, gps_log_summary, driver_statement, rider_history, policy_violation_note.",
    input_schema={
        "type": "object", "required": ["dispute_id", "response"],
        "properties": {
            "dispute_id": {"type": "integer"},
            "response": {"type": "object"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.submit_dispute_response(a["dispute_id"], a["response"]),
                              f"submitted dispute {a['dispute_id']}"),
)


# ----- safety ----- #

LIST_SAFETY_INCIDENTS = ToolSpec(
    name="list_safety_incidents",
    description="List safety incidents. Filter by severity_min (1=minor, 2=moderate, 3=severe).",
    input_schema={
        "type": "object",
        "properties": {"severity_min": {"type": "integer", "minimum": 1, "maximum": 3}},
    },
    handler=lambda sb, a: _ok(sb.rs.list_safety_incidents(**a), "incidents"),
)

GET_INCIDENT = ToolSpec(
    name="get_incident",
    description="Fetch a single safety incident with kind, severity, escalation level, evidence list.",
    input_schema={
        "type": "object", "required": ["incident_id"],
        "properties": {"incident_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.get_incident(a["incident_id"]),
                              f"incident {a['incident_id']}"),
)

ESCALATE_INCIDENT = ToolSpec(
    name="escalate_incident",
    description="Raise the escalation level of an incident (T1=1 minor, T2=2 moderate, T3=3 severe). Optionally notify both parties.",
    input_schema={
        "type": "object", "required": ["incident_id", "level"],
        "properties": {
            "incident_id": {"type": "integer"},
            "level": {"type": "integer", "minimum": 1, "maximum": 3},
            "notify_parties": {"type": "boolean", "default": True},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.escalate_incident(**a),
                              f"escalated {a['incident_id']}->T{a['level']}"),
)

CONTACT_EMERGENCY = ToolSpec(
    name="contact_emergency",
    description="Contact emergency services on behalf of an incident. kind defaults to 911.",
    input_schema={
        "type": "object", "required": ["incident_id"],
        "properties": {
            "incident_id": {"type": "integer"},
            "kind": {"type": "string", "default": "911"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.contact_emergency(**a),
                              f"emergency {a['incident_id']}"),
)

ATTACH_EVIDENCE = ToolSpec(
    name="attach_evidence",
    description="Attach evidence to a safety incident: gps_log, photo, audio, witness_statement, etc.",
    input_schema={
        "type": "object", "required": ["incident_id", "evidence_kind"],
        "properties": {
            "incident_id": {"type": "integer"},
            "evidence_kind": {"type": "string"},
            "note": {"type": "string", "default": ""},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.attach_evidence(**a),
                              f"evidence {a['incident_id']}"),
)


# ----- fraud ----- #

LIST_FLAGGED_TRIPS = ToolSpec(
    name="list_flagged_trips",
    description="List trips flagged for fraud (or with disputes attached).",
    input_schema={"type": "object", "properties": {}},
    handler=lambda sb, a: _ok(sb.rs.list_flagged_trips(), "flagged trips"),
)

LIST_FLAGGED_ACCOUNTS = ToolSpec(
    name="list_flagged_accounts",
    description="List flagged riders and drivers — anyone with a flag set on their account (frozen, fraud_suspected, etc.).",
    input_schema={"type": "object", "properties": {}},
    handler=lambda sb, a: _ok(sb.rs.list_flagged_accounts(), "flagged accounts"),
)

FREEZE_ACCOUNT = ToolSpec(
    name="freeze_account",
    description="Freeze a rider or driver account. target_kind is 'rider' or 'driver'.",
    input_schema={
        "type": "object", "required": ["target_kind", "target_id", "reason"],
        "properties": {
            "target_kind": {"type": "string", "enum": ["rider", "driver"]},
            "target_id": {"type": "integer"},
            "reason": {"type": "string"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.freeze_account(**a),
                              f"froze {a['target_kind']}:{a['target_id']}"),
)

BAN_DEVICE = ToolSpec(
    name="ban_device",
    description="Ban a device fingerprint platform-wide.",
    input_schema={
        "type": "object", "required": ["fingerprint", "reason"],
        "properties": {
            "fingerprint": {"type": "string"},
            "reason": {"type": "string"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.ban_device(**a),
                              f"banned device {a['fingerprint'][:8]}..."),
)

CLUSTER_BY_FINGERPRINT = ToolSpec(
    name="cluster_by_fingerprint",
    description="Find all riders, drivers, and trips that share the given device fingerprint. Powerful primitive for fraud-ring detection.",
    input_schema={
        "type": "object", "required": ["fingerprint"],
        "properties": {"fingerprint": {"type": "string"}},
    },
    handler=lambda sb, a: _ok(sb.rs.cluster_by_fingerprint(a["fingerprint"]),
                              "cluster"),
)


# ----- lost items ----- #

LIST_LOST_ITEMS = ToolSpec(
    name="list_lost_items",
    description="List reported lost items.",
    input_schema={
        "type": "object",
        "properties": {
            "trip_id": {"type": "integer"},
            "confirmed": {"type": "boolean"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.list_lost_items(**a), "lost items"),
)

CREATE_LOST_ITEM = ToolSpec(
    name="create_lost_item",
    description="Create a lost-item report on a trip with a description.",
    input_schema={
        "type": "object", "required": ["trip_id", "description"],
        "properties": {
            "trip_id": {"type": "integer"},
            "description": {"type": "string"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.create_lost_item(**a),
                              f"created lost item for {a['trip_id']}"),
)

ASSIGN_LOST_ITEM = ToolSpec(
    name="assign_lost_item",
    description="Assign a lost item to a driver (typically the original driver). Generates handoff_code.",
    input_schema={
        "type": "object", "required": ["lost_item_id", "driver_id"],
        "properties": {
            "lost_item_id": {"type": "integer"},
            "driver_id": {"type": "integer"},
            "return_method": {"type": "string", "default": "next_idle_window"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.assign_lost_item(**a),
                              f"assigned lost {a['lost_item_id']}"),
)

SCHEDULE_LOST_ITEM_PICKUP = ToolSpec(
    name="schedule_lost_item_pickup",
    description="Schedule the return-pickup time and place for a lost item. Sends notifications to both rider and driver with the handoff code.",
    input_schema={
        "type": "object",
        "required": ["lost_item_id", "pickup_at", "pickup_location"],
        "properties": {
            "lost_item_id": {"type": "integer"},
            "pickup_at": {"type": "number", "description": "Absolute simulated seconds"},
            "pickup_location": {"type": "array", "items": {"type": "number"},
                                  "minItems": 2, "maxItems": 2},
            "notify_rider": {"type": "boolean", "default": True},
            "notify_driver": {"type": "boolean", "default": True},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.schedule_lost_item_pickup(**a),
                              f"scheduled lost {a['lost_item_id']}"),
)


# ----- drivers + riders ----- #

GET_DRIVER = ToolSpec(
    name="get_driver",
    description="Fetch a driver record (location, status, rating, vehicle, docs, earnings, flags).",
    input_schema={
        "type": "object", "required": ["driver_id"],
        "properties": {"driver_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.get_driver(a["driver_id"]),
                              f"driver {a['driver_id']}"),
)

GET_DRIVER_DOCUMENTS = ToolSpec(
    name="get_driver_documents",
    description="Fetch a driver's compliance documents (drivers_license, vehicle_registration, background_check) with expiry dates.",
    input_schema={
        "type": "object", "required": ["driver_id"],
        "properties": {"driver_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.get_driver_documents(a["driver_id"]),
                              f"docs {a['driver_id']}"),
)

VERIFY_DRIVER_DOCUMENTS = ToolSpec(
    name="verify_driver_documents",
    description="Mark a driver's documents as verified, re-enabling them for dispatch.",
    input_schema={
        "type": "object", "required": ["driver_id"],
        "properties": {"driver_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.verify_driver_documents(a["driver_id"]),
                              f"verified {a['driver_id']}"),
)

ADJUST_DRIVER_PAYOUT = ToolSpec(
    name="adjust_driver_payout",
    description="Apply a payout adjustment to a driver — positive to credit them, negative to claw back.",
    input_schema={
        "type": "object", "required": ["driver_id", "amount", "reason"],
        "properties": {
            "driver_id": {"type": "integer"},
            "amount": {"type": "number"},
            "reason": {"type": "string"},
            "related_trip_id": {"type": "integer"},
        },
    },
    handler=lambda sb, a: _ok(
        sb.rs.adjust_driver_payout(
            a["driver_id"], amount=a["amount"], reason=a["reason"],
            related_trip_id=a.get("related_trip_id"),
        ),
        f"payout {a['driver_id']} {a['amount']:+.2f}",
    ),
)

GET_DRIVER_PERFORMANCE = ToolSpec(
    name="get_driver_performance",
    description="Driver's daily KPIs (rating, completed trips today, cancellations, earnings).",
    input_schema={
        "type": "object", "required": ["driver_id"],
        "properties": {"driver_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.get_driver_performance(a["driver_id"]),
                              f"perf {a['driver_id']}"),
)

GET_RIDER = ToolSpec(
    name="get_rider",
    description="Fetch a rider record (email, payment_bin, device_fingerprint, ratings, chargeback count, flags).",
    input_schema={
        "type": "object", "required": ["rider_id"],
        "properties": {"rider_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.get_rider(a["rider_id"]),
                              f"rider {a['rider_id']}"),
)

FREEZE_RIDER = ToolSpec(
    name="freeze_rider",
    description="Freeze a rider account. Reason should reference policy.",
    input_schema={
        "type": "object", "required": ["rider_id", "reason"],
        "properties": {
            "rider_id": {"type": "integer"},
            "reason": {"type": "string"},
        },
    },
    handler=lambda sb, a: _ok(
        sb.rs.freeze_rider(a["rider_id"], reason=a["reason"]),
        f"froze rider {a['rider_id']}",
    ),
)

RESTORE_RIDER = ToolSpec(
    name="restore_rider",
    description="Restore a frozen rider account, e.g. after positive identity verification.",
    input_schema={
        "type": "object", "required": ["rider_id"],
        "properties": {
            "rider_id": {"type": "integer"},
            "reason": {"type": "string", "default": ""},
        },
    },
    handler=lambda sb, a: _ok(
        sb.rs.restore_rider(a["rider_id"], reason=a.get("reason", "")),
        f"restored rider {a['rider_id']}",
    ),
)

LOGIN_HISTORY = ToolSpec(
    name="login_history",
    description="Fetch a rider's typical login pattern (device, zone, hour-window) plus any recent anomalies. Used for account-takeover detection.",
    input_schema={
        "type": "object", "required": ["rider_id"],
        "properties": {"rider_id": {"type": "integer"}},
    },
    handler=lambda sb, a: _ok(sb.rs.login_history(a["rider_id"]),
                              f"login history {a['rider_id']}"),
)


# ----- pricing + messaging ----- #

GET_PRICING_QUOTE = ToolSpec(
    name="get_pricing_quote",
    description="Quote a fare for a (pickup, dropoff) leg with a vehicle type. Includes current zone surge.",
    input_schema={
        "type": "object",
        "required": ["pickup_x", "pickup_y", "dropoff_x", "dropoff_y"],
        "properties": {
            "pickup_x": {"type": "number"},
            "pickup_y": {"type": "number"},
            "dropoff_x": {"type": "number"},
            "dropoff_y": {"type": "number"},
            "vehicle_type": {"type": "string", "default": "uberx"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.get_pricing_quote(**a), "quote"),
)

LIST_SURGE_ZONES = ToolSpec(
    name="list_surge_zones",
    description="List all zones with current surge multipliers.",
    input_schema={"type": "object", "properties": {}},
    handler=lambda sb, a: _ok(sb.rs.list_surge_zones(), "zones"),
)

SEND_TO_RIDER = ToolSpec(
    name="send_to_rider",
    description="Send a templated message (push/SMS) to a rider. Common templates: trip.refunded, lost_item.pickup_scheduled, account.frozen, safety.incident_update.",
    input_schema={
        "type": "object", "required": ["rider_id", "template"],
        "properties": {
            "rider_id": {"type": "integer"},
            "template": {"type": "string"},
            "variables": {"type": "object"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.send_to_rider(**a),
                              f"msg rider {a['rider_id']}"),
)

SEND_TO_DRIVER = ToolSpec(
    name="send_to_driver",
    description="Send a templated message to a driver. Common templates: lost_item.return_pickup, payout.adjusted, safety.incident_update.",
    input_schema={
        "type": "object", "required": ["driver_id", "template"],
        "properties": {
            "driver_id": {"type": "integer"},
            "template": {"type": "string"},
            "variables": {"type": "object"},
        },
    },
    handler=lambda sb, a: _ok(sb.rs.send_to_driver(**a),
                              f"msg driver {a['driver_id']}"),
)


# --------------------------------------------------------------------------- #
# Public registry — tasks pick subsets.
# --------------------------------------------------------------------------- #

ALL_TOOLS: dict[str, ToolSpec] = {
    spec.name: spec for spec in [
        # marketplace
        LIST_PENDING_REQUESTS, LIST_IDLE_DRIVERS, MATCH_RIDE, CANCEL_TRIP,
        AUTO_MATCH_NEAREST, SET_SURGE, REBALANCE_DRIVER, OFFER_INCENTIVE,
        ZONE_SUPPLY_DEMAND, DISPATCH_LOG, TICK,
        # trips
        LIST_TRIPS, GET_TRIP, GET_TRIP_GPS_LOG,
        # refunds + disputes
        CREATE_REFUND, LIST_REFUNDS, LIST_DISPUTES, GET_DISPUTE,
        SUBMIT_DISPUTE_RESPONSE,
        # safety
        LIST_SAFETY_INCIDENTS, GET_INCIDENT, ESCALATE_INCIDENT,
        CONTACT_EMERGENCY, ATTACH_EVIDENCE,
        # fraud
        LIST_FLAGGED_TRIPS, LIST_FLAGGED_ACCOUNTS, FREEZE_ACCOUNT,
        BAN_DEVICE, CLUSTER_BY_FINGERPRINT,
        # lost items
        LIST_LOST_ITEMS, CREATE_LOST_ITEM, ASSIGN_LOST_ITEM,
        SCHEDULE_LOST_ITEM_PICKUP,
        # drivers/riders
        GET_DRIVER, GET_DRIVER_DOCUMENTS, VERIFY_DRIVER_DOCUMENTS,
        ADJUST_DRIVER_PAYOUT, GET_DRIVER_PERFORMANCE,
        GET_RIDER, FREEZE_RIDER, RESTORE_RIDER, LOGIN_HISTORY,
        # pricing + messaging
        GET_PRICING_QUOTE, LIST_SURGE_ZONES, SEND_TO_RIDER, SEND_TO_DRIVER,
    ]
}


def select(*names: str) -> list[ToolSpec]:
    return [ALL_TOOLS[n] for n in names]
