"""Tests for non-blocking map generation in the triage loop."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMapOffload:
    """Verify map generation runs as a background task, not inline."""

    @pytest.mark.asyncio
    async def test_alert_queue_filled_before_map_starts(self):
        """
        The CorrelatedEvent should be on the alert_queue BEFORE
        map generation begins. We verify this by checking that
        _generate_and_upload_map is dispatched via asyncio.create_task.
        """
        import main

        with patch("main._generate_and_upload_map", new_callable=AsyncMock) as mock_map:
            with patch("main.asyncio.create_task") as mock_create_task:
                prediction = MagicMock()
                prediction.trajectory_points = [{"lat": 1, "lon": 2}]

                main._schedule_map_generation(
                    prediction, MagicMock(), "test-label",
                )

                mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_map_failure_does_not_affect_alert(self):
        """If map generation raises, the alert was already queued."""
        import main

        prediction = MagicMock()
        prediction.trajectory_points = [
            {"lat": 32.0, "lon": 34.0},
            {"lat": 32.1, "lon": 34.1},
        ]
        prediction.covariance_position_enu = [[1000, 0, 0], [0, 1000, 0], [0, 0, 1000]]
        prediction.impact_latitude = 32.1
        prediction.impact_longitude = 34.1
        prediction.seconds_until_impact = 120.0
        prediction.terminal_velocity_m_s = 800.0

        with (
            patch("main.render_ground_track", side_effect=RuntimeError("render failed")),
            patch("main.upload_slack_map", new_callable=AsyncMock),
        ):
            # Should not raise — exception caught internally
            await main._generate_and_upload_map(prediction, None, "test-label")
