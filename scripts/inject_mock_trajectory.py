#!/usr/bin/env python3
"""
Mock Trajectory Injector
=========================

Generates 4 synthetic TIP observations for a fake NORAD_CAT_ID and
POSTs them to the local /api/v1/test-event endpoint with a 2-second
delay between each.

Usage::

    python scripts/inject_mock_trajectory.py

Requires the local stack to be running (docker compose up).
"""

import random
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/api/v1/test-event"
NORAD_CAT_ID = "99999"

# Ground truth trajectory: re-entry over US Southwest
# Start: 37.0°N, 115.0°W, 120km altitude, heading SSW ~2 km/s
LAT0, LON0, ALT0 = 37.0, -115.0, 120_000.0
V_LAT = -0.006   # deg/s southward
V_LON = -0.003   # deg/s westward
V_ALT = -800.0   # m/s downward

OFFSETS_SEC = [0, 25, 55, 90]
NOISE_PROFILES = ["satellite", "thermal", "social_media", "thermal"]

# Noise standard deviations per profile (lat_deg, lon_deg, alt_m)
NOISE_SIGMA = {
    "satellite":    (0.002, 0.002, 400),
    "thermal":      (0.008, 0.008, 1500),
    "social_media": (0.020, 0.020, 4000),
}


def generate_observations() -> list[dict]:
    """Generate 4 noisy synthetic TIP observations."""
    random.seed(42)
    base_time = datetime.now(timezone.utc) - timedelta(seconds=max(OFFSETS_SEC))
    observations = []

    for i, t in enumerate(OFFSETS_SEC):
        true_lat = LAT0 + V_LAT * t
        true_lon = LON0 + V_LON * t
        true_alt = ALT0 + V_ALT * t + 0.5 * (-9.81) * t ** 2
        true_alt = max(true_alt, 0)

        profile = NOISE_PROFILES[i]
        sigma = NOISE_SIGMA[profile]
        noisy_lat = round(true_lat + random.gauss(0, sigma[0]), 6)
        noisy_lon = round(true_lon + random.gauss(0, sigma[1]), 6)
        noisy_alt = round(max(true_alt + random.gauss(0, sigma[2]), 500.0), 1)

        ts = base_time + timedelta(seconds=t)
        observations.append({
            "source": "spacetrack",
            "latitude": noisy_lat,
            "longitude": noisy_lon,
            "description": (
                f"TIP: NORAD {NORAD_CAT_ID} re-entry predicted "
                f"{ts.isoformat()} UTC @ ({noisy_lat}, {noisy_lon})"
            ),
            "NORAD_CAT_ID": NORAD_CAT_ID,
            "LAT": str(noisy_lat),
            "LON": str(noisy_lon),
            "ALTITUDE_M": str(noisy_alt),
            "DECAY_EPOCH": ts.isoformat(),
            "WINDOW": "5",
            "HIGH_INTEREST": "Y",
            "timestamp": ts.isoformat(),
        })

    return observations


def main() -> None:
    print("=" * 60)
    print("  MOCK TRAJECTORY INJECTOR")
    print("=" * 60)
    print(f"\n  Target:   {ENDPOINT}")
    print(f"  Object:   NORAD {NORAD_CAT_ID}")
    print(f"  Points:   {len(OFFSETS_SEC)} observations")
    print(f"  Delay:    2s between each POST\n")

    # Check health first
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        print(f"  Health check: {health.json()}\n")
    except httpx.ConnectError:
        print("  ERROR: Cannot connect to local server.")
        print("  Make sure 'docker compose up' is running.\n")
        sys.exit(1)

    observations = generate_observations()

    for i, obs in enumerate(observations):
        print(f"── Observation {i + 1}/{len(observations)} ──")
        print(f"  lat={obs['latitude']}, lon={obs['longitude']}, alt={obs['ALTITUDE_M']}m")
        print(f"  timestamp={obs['timestamp']}")
        print(f"  POSTing to {ENDPOINT}...")

        resp = httpx.post(ENDPOINT, json=obs, timeout=10.0)
        print(f"  Response: {resp.status_code} {resp.json()}")

        if i < len(observations) - 1:
            print(f"  Waiting 2 seconds...\n")
            time.sleep(2)
        else:
            print()

    print("=" * 60)
    print("  All observations injected.")
    print("  Check your Slack/Discord for impact prediction alerts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
