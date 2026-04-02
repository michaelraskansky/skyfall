# Ground Track Map Visualization Design

Render ImpactPrediction trajectory data as a static PNG map and upload to Slack.

## Map Generator (`visuals/map_generator.py`)

Function: `render_ground_track(prediction: ImpactPrediction, satcat_info: SatcatInfo | None) -> str`

Returns path to a temporary 720x480 PNG file.

**Rendering layers (staticmap):**
1. Red line: ground track from trajectory_points (lat, lon)
2. Blue circle: origin marker (first waypoint)
3. Red circle: impact marker (impact_latitude, impact_longitude)
4. Semi-transparent polygon: 95% confidence corridor — perpendicular buffer around ground track using covariance semi-major axis converted from metres to degrees

**File output:** `tempfile.NamedTemporaryFile(suffix='.png', delete=False)`. Caller cleans up.

**Non-blocking:** Called via `asyncio.to_thread()`.

## Slack Map Upload (`output/alerter.py`)

New function: `_upload_slack_map(file_path, caption)` using `slack_sdk.web.async_client.AsyncWebClient`.

**Config:** `slack_bot_token` and `slack_channel_id` in `config.py`. Empty = skip upload silently.

**Rate limiting:** Track last upload timestamp. Skip map upload if < 5 seconds since last. Text alerts are unaffected.

**Caption:** Object name, impact coordinates, ETA, terminal velocity, ellipse dimensions.

## Integration (`main.py`)

After predictor runs, inside existing try/except:
1. `map_path = await asyncio.to_thread(render_ground_track, prediction, satcat_info)`
2. `await _upload_slack_map(map_path, caption)`
3. `os.unlink(map_path)`

Map generation/upload failure does not block the text alert.

## Dependencies

- `staticmap` (pure Python, Pillow + requests)
- `slack_sdk`

## Files Changed

| File | Change |
|------|--------|
| `visuals/__init__.py` | New (empty) |
| `visuals/map_generator.py` | New — render_ground_track() |
| `config.py` | Add slack_bot_token, slack_channel_id |
| `output/alerter.py` | Add _upload_slack_map() with rate throttle |
| `main.py` | Call map generator + upload after prediction |
| `pyproject.toml` | Add staticmap, slack_sdk |
| `tests/unit/test_map_generator.py` | Unit tests |
