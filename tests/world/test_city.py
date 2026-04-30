"""City + zone + distance + travel-time tests."""

from __future__ import annotations

from rideshare_gym.world.city import default_city


def test_default_city_has_six_zones():
    city = default_city()
    assert city.name == "Bayview"
    assert len(city.zones) == 6
    ids = {z.id for z in city.zones}
    assert ids == {"downtown", "airport", "university", "stadium",
                    "suburb_n", "suburb_s"}


def test_distance_is_manhattan():
    city = default_city()
    assert city.distance_km((0.0, 0.0), (3.0, 4.0)) == 7.0


def test_travel_time_scales_with_distance_and_traffic():
    city = default_city()
    fast = city.travel_time_minutes((0.0, 0.0), (10.0, 0.0),
                                      traffic_factor=1.0)
    slow = city.travel_time_minutes((0.0, 0.0), (10.0, 0.0),
                                      traffic_factor=2.0)
    assert fast > 0
    assert slow > fast


def test_zone_for_returns_smallest_radius_zone_when_overlapping():
    city = default_city()
    z = city.zone_for(15.0, 15.0)
    assert z is not None
    assert z.id == "downtown"
