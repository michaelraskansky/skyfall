# Boost Phase EKF Design

Extend the `DebrisTrajectoryPredictor` to support ground-launched vehicles via a two-phase state machine: Boost → Ballistic.

## Problem

The EKF assumes all objects are already in orbit and descending. If the initial observation altitude is 0 (ground launch), the propagation loop terminates immediately because `alt <= 0` is true after the first step. This prevents tracking sub-orbital boosters and ballistic test flights.

## Model Changes (`trajectory/models.py`)

### TrajectoryRequest — new optional fields

```python
burn_time_seconds: Optional[float] = None      # default 60.0
thrust_to_mass_ratio: Optional[float] = None    # default 25.0 m/s²
pitch_angle_deg: Optional[float] = None         # default 85.0°
```

### ImpactPrediction — new field

```python
flight_phase_detected: str = "ballistic"  # "boost" or "ballistic"
```

## Predictor Changes (`trajectory/predictor.py`)

### Auto-detection in `predict()`

After sorting observations, inspect `obs[0].altitude_m`:
- If `< 5000m` → `flight_phase = "boost"`
- Otherwise → `flight_phase = "ballistic"` (existing behavior, no change)

### Parameter resolution

```python
burn_time = request.burn_time_seconds or 60.0
thrust = request.thrust_to_mass_ratio or 25.0
pitch_deg = request.pitch_angle_deg or 85.0
```

### `_ekf_predict(dt)` — thrust support

New parameter: `thrust_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)`.

Total acceleration: `a = gravity + drag + thrust_enu`.

Thrust is constant w.r.t. state variables, so it adds no extra Jacobian terms. The existing drag Jacobian is unchanged.

### `_propagate_to_impact()` — two-phase loop

1. Track `flight_time` accumulator (starts at 0).
2. If `flight_phase == "boost"` and `flight_time < burn_time`:
   - Compute thrust vector from pitch angle and heading.
   - Pass thrust_enu to `_ekf_predict(dt_step, thrust_enu=...)`.
3. Once `flight_time >= burn_time`: set `flight_phase = "ballistic"`, thrust becomes `(0, 0, 0)`.
4. Ground-impact termination: only check `alt <= 0` when phase is ballistic OR vertical velocity is negative (descending).

### Thrust vector decomposition

Vertical component: `thrust * sin(pitch_rad)`.
Horizontal component: `thrust * cos(pitch_rad)`, applied along the heading direction.

Heading is derived from the EKF's horizontal velocity vector `(v_e, v_n)`:
- `heading = (v_e, v_n) / ||(v_e, v_n)||`
- **Zero-velocity safeguard:** If horizontal speed < 0.1 m/s, default heading to `(1.0, 0.0)` (due east).

```python
horiz_speed = math.sqrt(ve**2 + vn**2)
if horiz_speed < 0.1:
    heading_e, heading_n = 1.0, 0.0
else:
    heading_e, heading_n = ve / horiz_speed, vn / horiz_speed

pitch_rad = math.radians(pitch_deg)
thrust_e = thrust * math.cos(pitch_rad) * heading_e
thrust_n = thrust * math.cos(pitch_rad) * heading_n
thrust_u = thrust * math.sin(pitch_rad)
```

## Files Changed

| File | Change |
|------|--------|
| `trajectory/models.py` | Add 3 optional fields to TrajectoryRequest, 1 field to ImpactPrediction |
| `trajectory/predictor.py` | Auto-detect phase, parameter resolution, thrust in `_ekf_predict`, two-phase loop in `_propagate_to_impact` |
| `trajectory/physics.py` | No changes — thrust is computed in the predictor, not the physics module |
| `tests/unit/test_predictor.py` | New tests for boost phase, ground launch, phase transition |
