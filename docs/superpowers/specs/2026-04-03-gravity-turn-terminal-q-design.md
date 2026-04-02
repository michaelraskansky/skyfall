# Gravity Turn & Terminal Q-Matrix Inflation Design

Replace the fixed-pitch thrust model with a kinematic gravity turn, and inflate process noise during terminal descent to reflect maneuver/turbulence uncertainty.

## 1. Gravity Turn (`_compute_thrust_enu`)

Replace the fixed pitch angle with a three-stage thrust program:

### Stage 1: Vertical Clear-Off
- Condition: `elapsed < clear_off_sec` (default 5.0s) OR `speed < 50 m/s`
- Thrust: pure vertical `(0, 0, thrust)`
- Purpose: clear the launch pad before pitching

### Stage 2: Pitch-Over Kick
- Trigger: one-time, at the moment clear-off ends
- Action: tilt thrust by `pitch_kick_deg` (default 2.0°) off vertical toward the heading
- Heading defaults to due east if horizontal speed < 0.1 m/s (existing safeguard)

### Stage 3: Gravity Turn
- Condition: remainder of burn after pitch-over
- Thrust: aligned exactly with the full 3D velocity vector `(ve, vn, vu)`
- Gravity naturally curves the trajectory into an efficient ballistic arc
- Zero-velocity safeguard: if total speed < 0.1 m/s, thrust straight up

### Method Signature
```python
_compute_thrust_enu(ve, vn, vu, thrust, pitch_kick_deg, elapsed, clear_off_sec=5.0)
```

### Parameter Reinterpretation
- `pitch_angle_deg` on `TrajectoryRequest` is reinterpreted as the pitch-over kick angle (degrees off vertical)
- Default changes from `None` (resolved to 85.0° from horizontal) to `None` (resolved to 2.0° off vertical)
- Backward compatible: ballistic phase never calls `_compute_thrust_enu`

## 2. Terminal Q-Matrix Inflation (`_propagate_to_impact`)

### Condition
All three must be true:
- `current_phase == "ballistic"`
- `alt < 30_000` meters
- `vu < 0` (descending)

### Scaling Function
```python
q_scale = 1.0 + (terminal_q_multiplier - 1.0) * (1.0 - alt / 30_000)
```
- At 30km: `q_scale = 1.0` (no inflation)
- At 15km: `q_scale ≈ 5.5`
- At 0m: `q_scale = terminal_q_multiplier`

### Default
`terminal_q_multiplier = 10.0` — configurable on `DebrisTrajectoryPredictor.__init__`, not per-request.

### Implementation
Add `q_scale: float = 1.0` parameter to `_make_Q(dt, q_scale)`. Multiply base `q_p` and `q_v` by `q_scale`. Pass computed scale from propagation loop.

### Effect
Wider 95% confidence ellipse at impact. Point estimate unchanged.

## Files Changed

| File | Change |
|------|--------|
| `trajectory/predictor.py` | Rewrite `_compute_thrust_enu`, add `q_scale` to `_make_Q`, add terminal inflation to propagation loop, add `terminal_q_multiplier` to `__init__` |
| `tests/unit/test_predictor.py` | Tests for gravity turn stages, terminal Q inflation, ellipse widening |
