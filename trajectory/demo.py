#!/usr/bin/env python3
"""
Trajectory Prediction Engine – Mock Execution Demo
====================================================

Simulates a high-velocity debris re-entry event by generating 4 noisy
"sensor pings" along a known ballistic arc, then feeds them into the
``DebrisTrajectoryPredictor`` and prints the predicted impact zone.

The simulated scenario
----------------------
A piece of orbital debris is re-entering over the US Southwest, travelling
roughly south-southwest.  It is first detected at ~120 km altitude over
southern Nevada and tracked through four observations as it descends
through the upper atmosphere.  The final observation is still at ~30 km
altitude, so the EKF must propagate the remaining trajectory to impact.

Run::

    python -m trajectory.demo
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

# Ensure the parent directory is importable when run as a script.
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from trajectory.models import SensorObservation, TrajectoryRequest
from trajectory.predictor import DebrisTrajectoryPredictor


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth trajectory generation (for the mock)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_true_trajectory() -> list[dict]:
    """
    Generate a "ground truth" ballistic arc for a debris fragment.

    We use simple kinematics in geodetic approximation:
      - Start: 37.0°N, 115.0°W, 120 km altitude.
      - Velocity: heading roughly SSW at ~2 km/s, descending at ~800 m/s.
      - Four observations at t = 0, 25, 55, 90 seconds.

    The last observation is still at ~30+ km altitude, forcing the EKF to
    propagate tens of seconds through the atmosphere to reach ground.
    """
    lat0, lon0, alt0 = 37.0, -115.0, 120_000.0  # metres

    # Approximate velocity components.
    # Total speed ≈ 2.2 km/s, mostly horizontal, moderate descent angle.
    v_lat = -0.006   # deg/s southward  (~670 m/s south)
    v_lon = -0.003   # deg/s westward   (~270 m/s west)
    v_alt = -800.0   # m/s downward (shallow re-entry)

    times = [0, 25, 55, 90]  # seconds after first detection
    truth = []
    for t in times:
        lat = lat0 + v_lat * t
        lon = lon0 + v_lon * t
        alt = alt0 + v_alt * t + 0.5 * (-9.81) * t ** 2
        alt = max(alt, 0)
        truth.append({"t": t, "lat": lat, "lon": lon, "alt": alt})

    return truth


def _add_noise(
    truth: list[dict],
    noise_profiles: list[str],
) -> list[SensorObservation]:
    """
    Take ground-truth waypoints and corrupt them with Gaussian noise
    appropriate to each sensor type.
    """
    noise_sigma = {
        "satellite":    (0.002, 0.002, 400),    # ~200 m horiz, 400 m vert
        "thermal":      (0.008, 0.008, 1_500),  # ~900 m horiz, 1.5 km vert
        "social_media": (0.020, 0.020, 4_000),  # ~2.2 km horiz, 4 km vert
        "default":      (0.012, 0.012, 2_500),
    }

    base_time = datetime(2026, 4, 1, 14, 30, 0, tzinfo=timezone.utc)
    observations = []

    for i, wp in enumerate(truth):
        profile = noise_profiles[i] if i < len(noise_profiles) else "default"
        sigma = noise_sigma.get(profile, noise_sigma["default"])

        noisy_lat = wp["lat"] + random.gauss(0, sigma[0])
        noisy_lon = wp["lon"] + random.gauss(0, sigma[1])
        noisy_alt = wp["alt"] + random.gauss(0, sigma[2])
        noisy_alt = max(noisy_alt, 500.0)  # clamp to reasonable minimum

        observations.append(SensorObservation(
            timestamp=base_time + timedelta(seconds=wp["t"]),
            latitude=round(noisy_lat, 6),
            longitude=round(noisy_lon, 6),
            altitude_m=round(noisy_alt, 1),
            noise_profile=profile,
        ))

    return observations


# ═══════════════════════════════════════════════════════════════════════════════
# Main demo
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    random.seed(42)
    np.random.seed(42)

    print("=" * 72)
    print("  DEBRIS TRAJECTORY PREDICTION ENGINE – V1 DEMO")
    print("=" * 72)

    # 1. Generate noisy observations
    truth = _generate_true_trajectory()
    profiles = ["satellite", "thermal", "social_media", "thermal"]
    observations = _add_noise(truth, profiles)

    print("\n── Simulated sensor pings (noisy) ─────────────────────────────\n")
    for i, ob in enumerate(observations):
        true_wp = truth[i]
        print(
            f"  Ping {i + 1} [{ob.noise_profile:>13}]  "
            f"t={true_wp['t']:>3}s  "
            f"lat={ob.latitude:>10.5f}  lon={ob.longitude:>11.5f}  "
            f"alt={ob.altitude_m:>9.1f} m"
        )
        print(
            f"  {'':>21}(true) "
            f"lat={true_wp['lat']:>10.5f}  lon={true_wp['lon']:>11.5f}  "
            f"alt={true_wp['alt']:>9.1f} m"
        )

    # 2. Build request
    request = TrajectoryRequest(
        object_id="DEBRIS-2026-04A",
        observations=observations,
        ballistic_coefficient=80.0,   # moderately dense fragment
        propagation_dt=0.5,           # 0.5-second propagation steps
    )

    # 3. Run predictor
    print("\n── Running EKF trajectory prediction ──────────────────────────\n")
    predictor = DebrisTrajectoryPredictor(
        ballistic_coefficient=request.ballistic_coefficient,
        process_noise_std_pos=100.0,
        process_noise_std_vel=20.0,
    )
    result = predictor.predict(request)

    # 4. Print results
    print("── PREDICTION RESULTS ─────────────────────────────────────────\n")
    print(f"  Object ID:            {result.object_id}")
    print(f"  Predicted impact:     ({result.impact_latitude:.5f}, {result.impact_longitude:.5f})")
    print(f"  Time of impact (UTC): {result.time_of_impact_utc.isoformat()}")
    print(f"  Seconds until impact: {result.seconds_until_impact:.1f} s")
    print(f"  Terminal velocity:    {result.terminal_velocity_m_s:.1f} m/s")
    print(f"  Final state (ENU):    {result.filter_state_at_impact}")

    # Covariance → 1-σ position uncertainty
    cov = np.array(result.covariance_position_enu)
    sigmas = np.sqrt(np.diag(cov))
    print(f"\n  Position 1-σ uncertainty at impact:")
    print(f"    East:  {sigmas[0]:>10.1f} m")
    print(f"    North: {sigmas[1]:>10.1f} m")
    print(f"    Up:    {sigmas[2]:>10.1f} m")

    # Approximate 95% confidence ellipse semi-axes (2-σ of the horizontal block)
    cov_horiz = cov[:2, :2]
    eigvals = np.linalg.eigvalsh(cov_horiz)
    semi_axes = 2.0 * np.sqrt(np.maximum(eigvals, 0))  # 95% ≈ 2σ
    print(f"\n  95% confidence ellipse semi-axes (horizontal):")
    print(f"    Major: {max(semi_axes):>10.1f} m  ({max(semi_axes)/1000:.1f} km)")
    print(f"    Minor: {min(semi_axes):>10.1f} m  ({min(semi_axes)/1000:.1f} km)")

    # Trajectory sample
    n_wps = len(result.trajectory_points)
    print(f"\n  Trajectory waypoints ({n_wps} sampled):")
    # Show first 5 and last 3.
    show = result.trajectory_points[:5]
    if n_wps > 8:
        show.append({"_ellipsis": True})
        show.extend(result.trajectory_points[-3:])
    elif n_wps > 5:
        show.extend(result.trajectory_points[5:])

    for wp in show:
        if "_ellipsis" in wp:
            print(f"    ... ({n_wps - 8} more waypoints)")
            continue
        print(
            f"    t={wp['t_sec']:>7.1f}s  "
            f"({wp['lat']:>10.5f}, {wp['lon']:>11.5f})  "
            f"alt={wp['alt_m']:>9.1f} m  "
            f"speed={wp['speed_m_s']:>8.1f} m/s"
        )

    # Full JSON output (truncated trajectory for readability)
    print("\n── Full JSON response (trajectory truncated) ───────────────────\n")
    output = result.model_dump(mode="json")
    output["trajectory_points"] = output["trajectory_points"][:3]
    if n_wps > 3:
        output["trajectory_points"].append({"_note": f"... {n_wps - 3} more waypoints"})
    print(json.dumps(output, indent=2, default=str))

    print("\n" + "=" * 72)
    print("  Demo complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
