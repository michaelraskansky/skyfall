#!/usr/bin/env python3
"""
Mock Trajectory Injector
=========================

Simulates a realistic re-entry detection scenario:

1. Injects a Space-Track TIP alert (seeds the object tracker, but does NOT
   trigger the EKF — TIPs are predictions, not sensor observations).
2. Injects 3 synthetic sensor observations (FIRMS thermal) that simulate
   ground-based detection of the re-entering object.  These DO trigger the
   EKF once ≥2 non-TIP observations are present.

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

# Observation 0 = TIP (spacetrack), observations 1-3 = sensor (firms)
SOURCES = ["spacetrack", "firms", "firms", "firms"]
NOISE_PROFILES = ["satellite", "thermal", "thermal", "thermal"]

# Noise standard deviations per profile (lat_deg, lon_deg, alt_m)
NOISE_SIGMA = {
    "satellite":    (0.002, 0.002, 400),
    "thermal":      (0.008, 0.008, 1500),
}


def generate_observations() -> list[dict]:
    """Generate 1 TIP + 3 sensor observations along a re-entry trajectory."""
    random.seed(42)
    base_time = datetime.now(timezone.utc) - timedelta(seconds=max(OFFSETS_SEC))
    observations = []

    for i, t in enumerate(OFFSETS_SEC):
        true_lat = LAT0 + V_LAT * t
        true_lon = LON0 + V_LON * t
        true_alt = ALT0 + V_ALT * t + 0.5 * (-9.81) * t ** 2
        true_alt = max(true_alt, 0)

        profile = NOISE_PROFILES[i]
        sigma = NOISE_SIGMA.get(profile, NOISE_SIGMA["thermal"])
        noisy_lat = round(true_lat + random.gauss(0, sigma[0]), 6)
        noisy_lon = round(true_lon + random.gauss(0, sigma[1]), 6)
        noisy_alt = round(max(true_alt + random.gauss(0, sigma[2]), 500.0), 1)

        ts = base_time + timedelta(seconds=t)
        source = SOURCES[i]

        obs = {
            "source": source,
            "latitude": noisy_lat,
            "longitude": noisy_lon,
            "description": (
                f"{'TIP' if source == 'spacetrack' else 'FIRMS thermal'}: "
                f"NORAD {NORAD_CAT_ID} @ ({noisy_lat}, {noisy_lon})"
            ),
            "NORAD_CAT_ID": NORAD_CAT_ID,
            "LAT": str(noisy_lat),
            "LON": str(noisy_lon),
            "ALTITUDE_M": str(noisy_alt),
        }

        if source == "spacetrack":
            obs["DECAY_EPOCH"] = ts.isoformat()
            obs["MSG_EPOCH"] = ts.strftime("%Y-%m-%d %H:%M:%S")
            obs["WINDOW"] = "5"
            obs["HIGH_INTEREST"] = "Y"

        observations.append(obs)

    return observations


def main() -> None:
    print("=" * 60)
    print("  MOCK TRAJECTORY INJECTOR")
    print("=" * 60)
    print(f"\n  Target:   {ENDPOINT}")
    print(f"  Object:   NORAD {NORAD_CAT_ID}")
    print(f"  Scenario: 1 TIP alert + 3 FIRMS sensor observations")
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
        source = obs["source"]
        label = "TIP (no EKF trigger)" if source == "spacetrack" else "FIRMS sensor"
        print(f"── Observation {i + 1}/{len(observations)} [{label}] ──")
        print(f"  source={source}, lat={obs['latitude']}, lon={obs['longitude']}, alt={obs.get('ALTITUDE_M', '?')}m")
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
    print("  Expected: No EKF on obs 1 (TIP) or 2 (only 1 sensor).")
    print("  EKF triggers on obs 3 (2 sensors) and 4 (3 sensors).")
    print("  Check Slack/Discord for impact prediction alerts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
