"""
Debris Trajectory Predictor – Extended Kalman Filter (V1)
==========================================================

This is the core class of the trajectory prediction engine.  It uses an
Extended Kalman Filter (EKF) to fuse sparse, noisy sensor observations
into a smooth 6-DOF state estimate, then propagates the ballistic
trajectory forward until the debris reaches ground level (altitude ≤ 0).

Design decisions for V1
-----------------------
* **No ML** – pure Newtonian mechanics + EKF.
* **No heavy frameworks** – the EKF is implemented directly with numpy
  (no filterpy dependency) for maximum portability.
* **Local ENU frame** – all filter math is done in a flat East-North-Up
  Cartesian frame anchored at the first observation.  This avoids the
  numerical headaches of filtering in geodetic coordinates directly.
* **Nonlinear state transition** – atmospheric drag makes the dynamics
  nonlinear, so we use the *Extended* KF (linearise F around the current
  state at each step).
* **Configurable measurement noise** – the ``noise_profile`` on each
  observation selects a preset R matrix so that satellite pings are
  trusted much more than social-media reports.

State vector (6 elements)
--------------------------
    x = [e, n, u, v_e, v_n, v_u]

    e, n, u   – position in local ENU (metres)
    v_e, v_n, v_u – velocity in local ENU (m/s)

Measurement vector (3 elements)
-------------------------------
    z = [e_obs, n_obs, u_obs]

    Observed position in local ENU, converted from the incoming geodetic
    (lat, lon, alt) via the coordinate transforms in ``physics.py``.

Usage
-----
::

    predictor = DebrisTrajectoryPredictor(ballistic_coefficient=80.0)
    result = predictor.predict(trajectory_request)
    # result is an ImpactPrediction with lat, lon, time, covariance, etc.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajectory.models import (
    ImpactPrediction,
    SensorObservation,
    TrajectoryRequest,
)
from trajectory.physics import (
    GRAVITY,
    air_density,
    drag_acceleration,
    enu_to_geodetic,
    geodetic_to_enu,
    gravity_acceleration,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight EKF state container (replaces filterpy dependency)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EKFState:
    """
    Minimal container holding the two things an EKF tracks:
      - x : state mean vector       (dim_x,)
      - P : state covariance matrix  (dim_x, dim_x)
    """

    x: np.ndarray  # (6,)
    P: np.ndarray  # (6, 6)


# ═══════════════════════════════════════════════════════════════════════════════
# Noise profiles  (diagonal σ values in metres for position)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each profile specifies the 1-σ standard deviation in metres for the
# (east, north, up) position measurement.  The R matrix is diag(σ²).

NOISE_PROFILES: Dict[str, Tuple[float, float, float]] = {
    "satellite":     (200.0,   200.0,   500.0),    # satellite thermal: ±200 m horiz
    "thermal":       (1_000.0, 1_000.0, 2_000.0),  # ground thermal sensor: ±1 km
    "social_media":  (5_000.0, 5_000.0, 10_000.0), # social media report: ±5 km
    "default":       (2_000.0, 2_000.0, 4_000.0),  # middle ground
}


def _make_R(profile: str) -> np.ndarray:
    """Build a 3×3 measurement-noise covariance matrix for the given profile."""
    sigma = NOISE_PROFILES.get(profile, NOISE_PROFILES["default"])
    return np.diag([s ** 2 for s in sigma])


# ═══════════════════════════════════════════════════════════════════════════════
# Predictor class
# ═══════════════════════════════════════════════════════════════════════════════


class DebrisTrajectoryPredictor:
    """
    EKF-based ballistic trajectory predictor for atmospheric re-entry debris.

    Parameters
    ----------
    ballistic_coefficient : float
        β = m / (C_d · A) in kg/m².  Controls how strongly drag decelerates
        the object.  Typical space debris: 20–150 kg/m².
    process_noise_std_pos : float
        1-σ process noise injected per second into position states (m).
        Accounts for unmodelled forces (wind, tumbling, ablation).
    process_noise_std_vel : float
        1-σ process noise injected per second into velocity states (m/s).
    """

    def __init__(
        self,
        ballistic_coefficient: float = 50.0,
        process_noise_std_pos: float = 50.0,
        process_noise_std_vel: float = 10.0,
    ) -> None:
        self.beta: float = ballistic_coefficient
        self.q_pos: float = process_noise_std_pos
        self.q_vel: float = process_noise_std_vel

        # ENU reference origin – set from the first observation.
        self._ref_lat: Optional[float] = None
        self._ref_lon: Optional[float] = None
        self._ref_alt: Optional[float] = None

        # EKF state (created during predict()).
        self._ekf: Optional[EKFState] = None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, request: TrajectoryRequest) -> ImpactPrediction:
        """
        Run the full pipeline: initialise → update with observations →
        propagate to impact → return prediction.

        Parameters
        ----------
        request : TrajectoryRequest
            Contains the ordered observations and tuning parameters.

        Returns
        -------
        ImpactPrediction with impact coordinates, time, velocity, covariance.
        """
        self.beta = request.ballistic_coefficient
        obs = sorted(request.observations, key=lambda o: o.timestamp)

        # ── 1. Set ENU reference origin from the first observation ────────
        self._ref_lat = obs[0].latitude
        self._ref_lon = obs[0].longitude
        self._ref_alt = obs[0].altitude_m

        # ── 2. Initialise the EKF ────────────────────────────────────────
        self._init_ekf(obs)

        # ── 3. Process each observation (predict-update cycle) ───────────
        last_time = obs[0].timestamp
        for i, ob in enumerate(obs):
            if i == 0:
                # First observation was used for initialisation.
                continue

            dt = (ob.timestamp - last_time).total_seconds()
            if dt <= 0:
                continue

            # --- EKF PREDICT step (nonlinear state transition) ---
            self._ekf_predict(dt)

            # --- EKF UPDATE step (linear measurement model) ---
            z = self._observation_to_enu(ob)
            R = _make_R(ob.noise_profile)
            self._ekf_update(z, R)

            last_time = ob.timestamp

        # ── 4. Forward-propagate until altitude ≤ 0 ─────────────────────
        impact_state, impact_cov, impact_dt, trajectory = self._propagate_to_impact(
            dt_step=request.propagation_dt,
            last_obs_time=last_time,
        )

        # ── 5. Convert results back to geodetic ─────────────────────────
        e, n, u = impact_state[0], impact_state[1], impact_state[2]
        vx, vy, vz = impact_state[3], impact_state[4], impact_state[5]
        impact_lat, impact_lon, impact_alt = enu_to_geodetic(
            e, n, u, self._ref_lat, self._ref_lon, self._ref_alt
        )
        terminal_speed = float(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))

        # Extract the 3×3 position covariance block (top-left of 6×6).
        pos_cov = impact_cov[:3, :3].tolist()

        impact_time = last_time + timedelta(seconds=impact_dt)

        return ImpactPrediction(
            object_id=request.object_id,
            impact_latitude=round(impact_lat, 6),
            impact_longitude=round(impact_lon, 6),
            impact_altitude_m=round(impact_alt, 1),
            time_of_impact_utc=impact_time,
            seconds_until_impact=round(impact_dt, 2),
            terminal_velocity_m_s=round(terminal_speed, 2),
            covariance_position_enu=pos_cov,
            trajectory_points=trajectory,
            filter_state_at_impact=[round(float(v), 4) for v in impact_state],
        )

    # ──────────────────────────────────────────────────────────────────────
    # EKF Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_ekf(self, obs: List[SensorObservation]) -> None:
        """
        Create and initialise the Extended Kalman Filter.

        The initial state is derived from the first two observations:
        position from obs[0], velocity estimated from the displacement
        between obs[0] and obs[1].
        """
        # --- Initial position (obs[0] is the ENU origin → [0, 0, 0]) ---
        p0 = np.array([0.0, 0.0, 0.0])

        # --- Initial velocity from first two observations ---
        p1 = np.array(self._observation_to_enu(obs[1]))
        dt01 = max((obs[1].timestamp - obs[0].timestamp).total_seconds(), 0.1)
        v0 = (p1 - p0) / dt01

        x = np.array([p0[0], p0[1], p0[2], v0[0], v0[1], v0[2]])

        # --- Initial covariance: high uncertainty ---
        # Position uncertainty: ~5 km (we trust the first ping loosely).
        # Velocity uncertainty: ~2 km/s (re-entry speeds are 1–8 km/s).
        P = np.diag([
            5_000.0 ** 2,   # e  (m²)
            5_000.0 ** 2,   # n
            10_000.0 ** 2,  # u  (altitude is noisier)
            2_000.0 ** 2,   # ve (m/s)²
            2_000.0 ** 2,   # vn
            2_000.0 ** 2,   # vu
        ])

        self._ekf = EKFState(x=x, P=P)

        logger.info(
            "EKF initialised: pos=[%.0f, %.0f, %.0f] m, "
            "vel=[%.0f, %.0f, %.0f] m/s",
            *self._ekf.x[:3], *self._ekf.x[3:6],
        )

    # ──────────────────────────────────────────────────────────────────────
    # EKF Predict (nonlinear state transition)
    # ──────────────────────────────────────────────────────────────────────

    def _ekf_predict(self, dt: float) -> None:
        """
        Propagate the state forward by *dt* seconds using nonlinear dynamics
        (gravity + drag), and update the covariance via the Jacobian F.

        State transition (Euler integration):

            e'  = e  + ve·dt + ½·ae·dt²
            n'  = n  + vn·dt + ½·an·dt²
            u'  = u  + vu·dt + ½·au·dt²
            ve' = ve + ae·dt
            vn' = vn + an·dt
            vu' = vu + au·dt

        where a = a_gravity + a_drag.
        """
        ekf = self._ekf
        x = ekf.x.copy()

        e, n, u = x[0], x[1], x[2]
        ve, vn, vu = x[3], x[4], x[5]

        # Current altitude in geodetic (approximate: u + ref_alt).
        alt = u + self._ref_alt

        # Total acceleration = gravity + drag
        g = gravity_acceleration()  # (0, 0, -9.81)
        d = drag_acceleration(ve, vn, vu, alt, self.beta)
        ae = g[0] + d[0]
        an = g[1] + d[1]
        au = g[2] + d[2]

        # --- Propagate state (Euler) ---
        x_new = np.array([
            e  + ve * dt + 0.5 * ae * dt ** 2,
            n  + vn * dt + 0.5 * an * dt ** 2,
            u  + vu * dt + 0.5 * au * dt ** 2,
            ve + ae * dt,
            vn + an * dt,
            vu + au * dt,
        ])

        # --- Jacobian of the state transition (F = ∂f/∂x) ---
        # For the drag term, we need partial derivatives of a_drag w.r.t.
        # velocity components.  Analytically:
        #
        #   a_drag_i = -½ρ/β · |v| · v_i
        #
        # ∂a_drag_i/∂v_j = -½ρ/β · (δ_ij · |v| + v_i · v_j / |v|)
        #
        # where δ_ij is the Kronecker delta.

        speed = math.sqrt(ve ** 2 + vn ** 2 + vu ** 2)
        rho = air_density(alt)
        k = 0.5 * rho / self.beta  # common factor

        F = np.eye(6)
        # Position rows: dp/dp = I, dp/dv = I·dt  (+ small drag correction)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        if speed > 1e-3:
            # Partial derivatives of drag acceleration w.r.t. velocity
            # da_i/dv_j = -k * (delta_ij * speed + v_i * v_j / speed)
            v_vec = np.array([ve, vn, vu])
            outer = np.outer(v_vec, v_vec) / speed
            da_dv = -k * (np.eye(3) * speed + outer)

            # Velocity rows: dv'/dv = I + da_dv · dt
            F[3:6, 3:6] = np.eye(3) + da_dv * dt

            # Position rows get a second-order correction: dp'/dv = I·dt + ½·da_dv·dt²
            F[0:3, 3:6] = np.eye(3) * dt + 0.5 * da_dv * dt ** 2

        # --- Process noise Q (continuous white noise acceleration model) ---
        Q = self._make_Q(dt)

        # --- Covariance propagation: P' = F P F^T + Q ---
        ekf.P = F @ ekf.P @ F.T + Q
        ekf.x = x_new

    # ──────────────────────────────────────────────────────────────────────
    # EKF Update (linear measurement model)
    # ──────────────────────────────────────────────────────────────────────

    def _ekf_update(self, z: Tuple[float, float, float], R: np.ndarray) -> None:
        """
        Incorporate a position measurement z = [e, n, u] into the filter.

        The measurement model is linear:

            z = H · x + noise

        where H = [I_3×3 | 0_3×3]  (we observe position, not velocity).
        """
        ekf = self._ekf
        z_arr = np.array(z)

        # Measurement matrix: we observe position (first 3 states), not velocity.
        H = np.zeros((3, 6))
        H[0, 0] = 1.0  # e
        H[1, 1] = 1.0  # n
        H[2, 2] = 1.0  # u

        # Innovation (measurement residual)
        y = z_arr - H @ ekf.x

        # Innovation covariance
        S = H @ ekf.P @ H.T + R

        # Kalman gain
        K = ekf.P @ H.T @ np.linalg.inv(S)

        # State update
        ekf.x = ekf.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(6) - K @ H
        ekf.P = I_KH @ ekf.P @ I_KH.T + K @ R @ K.T

        logger.debug(
            "EKF update: pos=[%.0f, %.0f, %.0f], vel=[%.0f, %.0f, %.0f]",
            *ekf.x[:3], *ekf.x[3:6],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Forward propagation to ground impact
    # ──────────────────────────────────────────────────────────────────────

    def _propagate_to_impact(
        self,
        dt_step: float,
        last_obs_time: datetime,
        max_propagation_sec: float = 3600.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, List[dict]]:
        """
        Propagate the filtered state forward in time until altitude ≤ 0
        or the maximum propagation time is exceeded.

        When the altitude crosses zero between two steps, a bisection
        refinement narrows the impact moment to within 0.01 s, preventing
        gross overshoot at high velocities.

        Returns
        -------
        state      : 6-element state vector at impact.
        covariance : 6×6 covariance matrix at impact.
        elapsed    : Total propagation time in seconds.
        trajectory : List of sampled waypoints for visualisation.
        """
        ekf = self._ekf
        elapsed = 0.0
        trajectory: List[dict] = []

        # Sample trajectory for visualization every N steps.
        sample_interval = max(1, int(10.0 / dt_step))
        step_count = 0

        while elapsed < max_propagation_sec:
            alt = ekf.x[2] + self._ref_alt

            # Record trajectory waypoint periodically.
            if step_count % sample_interval == 0:
                lat, lon, alt_geo = enu_to_geodetic(
                    float(ekf.x[0]), float(ekf.x[1]), float(ekf.x[2]),
                    self._ref_lat, self._ref_lon, self._ref_alt,
                )
                trajectory.append({
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "alt_m": round(alt_geo, 1),
                    "t_sec": round(elapsed, 2),
                    "speed_m_s": round(float(np.linalg.norm(ekf.x[3:6])), 1),
                })

            # Check termination: altitude ≤ 0.
            if alt <= 0 and elapsed > 0:
                logger.info(
                    "Impact predicted at t+%.1f s, alt=%.0f m", elapsed, alt
                )
                break

            # Save pre-step state for bisection if needed.
            prev_x = ekf.x.copy()
            prev_P = ekf.P.copy()
            prev_alt = alt

            # Propagate one step.
            self._ekf_predict(dt_step)
            elapsed += dt_step
            step_count += 1

            new_alt = ekf.x[2] + self._ref_alt

            # ── Bisection refinement when altitude crosses zero ──────────
            # If we went from positive to negative in one step, roll back
            # and use bisection to find the precise crossing.
            if prev_alt > 0 and new_alt <= 0:
                ekf.x = prev_x
                ekf.P = prev_P
                elapsed -= dt_step

                # Bisect within this time step to find t where alt ≈ 0.
                lo, hi = 0.0, dt_step
                for _ in range(20):  # ~0.001s precision after 20 iterations
                    mid = (lo + hi) / 2.0
                    # Temporarily propagate by mid seconds.
                    saved_x = ekf.x.copy()
                    saved_P = ekf.P.copy()
                    self._ekf_predict(mid)
                    mid_alt = ekf.x[2] + self._ref_alt
                    ekf.x = saved_x
                    ekf.P = saved_P
                    if mid_alt > 0:
                        lo = mid
                    else:
                        hi = mid

                # Final propagation to the refined impact time.
                self._ekf_predict(hi)
                elapsed += hi

                logger.info(
                    "Impact refined via bisection at t+%.2f s, alt=%.1f m",
                    elapsed, ekf.x[2] + self._ref_alt,
                )
                break
        else:
            logger.warning(
                "Max propagation time (%.0f s) exceeded – no impact found.",
                max_propagation_sec,
            )

        return ekf.x.copy(), ekf.P.copy(), elapsed, trajectory

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _observation_to_enu(
        self, ob: SensorObservation
    ) -> Tuple[float, float, float]:
        """Convert a geodetic observation to local ENU metres."""
        return geodetic_to_enu(
            ob.latitude, ob.longitude, ob.altitude_m,
            self._ref_lat, self._ref_lon, self._ref_alt,
        )

    def _make_Q(self, dt: float) -> np.ndarray:
        """
        Build the 6×6 process-noise covariance matrix for time step *dt*.

        Uses the piece-wise constant white-noise acceleration model:

            Q = G · diag(σ²) · G^T

        where G is the noise-gain matrix mapping acceleration noise to
        position/velocity states:

            G = [[½dt²  0     0   ]
                 [0      ½dt²  0   ]
                 [0      0     ½dt²]
                 [dt     0     0   ]
                 [0      dt    0   ]
                 [0      0     dt  ]]
        """
        q_p = self.q_pos ** 2
        q_v = self.q_vel ** 2

        # Simplified block-diagonal Q:
        #   Position block: σ_pos² · dt  (random walk)
        #   Velocity block: σ_vel² · dt  (random walk in velocity)
        #   Cross-terms:    σ_vel² · ½dt²
        Q = np.zeros((6, 6))

        # Position variance grows with dt (driven by velocity noise)
        Q[0, 0] = q_p * dt + q_v * dt ** 3 / 3.0
        Q[1, 1] = q_p * dt + q_v * dt ** 3 / 3.0
        Q[2, 2] = q_p * dt + q_v * dt ** 3 / 3.0

        # Velocity variance
        Q[3, 3] = q_v * dt
        Q[4, 4] = q_v * dt
        Q[5, 5] = q_v * dt

        # Cross-covariance position–velocity
        cross = q_v * dt ** 2 / 2.0
        Q[0, 3] = Q[3, 0] = cross
        Q[1, 4] = Q[4, 1] = cross
        Q[2, 5] = Q[5, 2] = cross

        return Q
