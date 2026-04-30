"""Fraud-engine tests — ring planted with shared signals."""

from __future__ import annotations

from rideshare_gym.world.fraud_engine import detect_account_farm_signals, plant_fraud_ring
from rideshare_gym.world.world import World


def test_account_farm_ring_has_shared_fingerprint_and_bin():
    w = World(tenant_id="t1")
    ring = plant_fraud_ring(w, kind="account_farm", n_riders=5, seed=1)
    assert len(ring.rider_ids) == 5
    fps = {w.riders[rid].device_fingerprint for rid in ring.rider_ids}
    bins = {w.riders[rid].payment_bin for rid in ring.rider_ids}
    assert len(fps) == 1
    assert len(bins) == 1


def test_obvious_count_controls_visible_flagging():
    w = World(tenant_id="t1")
    ring = plant_fraud_ring(w, kind="account_farm", n_riders=5, seed=2,
                              obvious_count=3)
    obvious = [w.riders[rid] for rid in ring.rider_ids
                if "high_risk_signup" in w.riders[rid].flags]
    assert len(obvious) == 3


def test_collusion_ring_has_drivers_and_riders():
    w = World(tenant_id="t1")
    ring = plant_fraud_ring(w, kind="collusion_ring", n_riders=5, seed=3)
    assert len(ring.driver_ids) == 3
    assert len(ring.rider_ids) == 5
    # All share fingerprint
    rider_fps = {w.riders[rid].device_fingerprint for rid in ring.rider_ids}
    driver_fps = {w.drivers[did].device_fingerprint for did in ring.driver_ids}
    assert rider_fps == driver_fps


def test_signals_helper_surfaces_disposable_bin_and_email():
    w = World(tenant_id="t1")
    ring = plant_fraud_ring(w, kind="account_farm", n_riders=2, seed=4)
    rider = w.riders[ring.rider_ids[0]]
    sigs = detect_account_farm_signals(rider)
    assert "disposable_bin" in sigs
    assert "disposable_email" in sigs
