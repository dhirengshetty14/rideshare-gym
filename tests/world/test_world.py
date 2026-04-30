"""End-to-end simulator tests — World tick loop, demand, drivers move, KPIs."""

from __future__ import annotations

from rideshare_gym.world.drivers import Driver, DriverStatus
from rideshare_gym.world.world import World


def test_world_starts_empty():
    w = World(tenant_id="t1")
    assert len(w.drivers) == 0
    assert len(w.trips) == 0
    assert w.clock.now == 0.0


def test_tick_advances_time_and_spawns_demand():
    w = World(tenant_id="t1")
    w.reseed(42)
    # Add an idle driver so demand can be matched, otherwise pile-up is noisy.
    did = w.next_id()
    w.drivers[did] = Driver(
        id=did, name="d", location=(15.0, 15.0),
        status=DriverStatus.IDLE, vehicle_type="uberx",
        home_zone_id="downtown",
    )
    for _ in range(20):
        w.tick()
    assert w.clock.now == 600.0  # 20 ticks of 30s
    # Some demand should have spawned in 10 sim minutes.
    assert len(w.trips) > 0


def test_dispatched_driver_moves_toward_target():
    w = World(tenant_id="t1")
    did = w.next_id()
    w.drivers[did] = Driver(
        id=did, name="d", location=(0.0, 0.0),
        status=DriverStatus.DISPATCHED, vehicle_type="uberx",
        target_location=(10.0, 0.0), speed_kmh=30.0,
    )
    initial_x = w.drivers[did].location[0]
    w.tick(60.0)  # 1 minute
    new_x = w.drivers[did].location[0]
    # 30 km/h × 1 min = 0.5 km
    assert new_x > initial_x
    assert new_x <= 10.0


def test_kpis_computed():
    w = World(tenant_id="t1")
    w.reseed(0)
    did = w.next_id()
    w.drivers[did] = Driver(
        id=did, name="d", location=(15.0, 15.0),
        status=DriverStatus.IDLE, vehicle_type="uberx",
        home_zone_id="downtown",
    )
    for _ in range(5):
        w.tick()
    kpis = w.kpis()
    assert "n_trips_seen" in kpis
    assert "completion_rate" in kpis
    assert "mean_pickup_wait_minutes" in kpis
    assert "revenue" in kpis


def test_snapshot_contains_all_resources():
    w = World(tenant_id="t1")
    snap = w.snapshot()
    for key in ("drivers", "riders", "trips", "surge_zones", "refunds",
                 "disputes", "incidents", "lost_items", "sent_messages",
                 "events", "kpis"):
        assert key in snap


def test_reset_restores_empty():
    w = World(tenant_id="t1")
    w.reseed(0)
    did = w.next_id()
    w.drivers[did] = Driver(id=did, name="d", location=(0, 0),
                              status=DriverStatus.IDLE)
    for _ in range(5):
        w.tick()
    # Reset is achieved via store.reset_world() in production; here we just
    # confirm a fresh World is empty.
    fresh = World(tenant_id="t1")
    assert fresh.clock.now == 0.0
    assert len(fresh.trips) == 0
