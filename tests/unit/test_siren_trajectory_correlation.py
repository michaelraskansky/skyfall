"""Unit tests for bidirectional siren-trajectory correlation in main.py."""

import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest


class TestPendingSirens:
    """Tests for the _pending_sirens reverse-lookup."""

    @pytest.mark.asyncio
    async def test_siren_stored_in_pending(self):
        """When a siren fires with no trajectory match, it is stored in _pending_sirens."""
        import main

        # Reset module-level state
        main._pending_sirens.clear()
        main._recent_predictions.clear()

        from ingestion.siren_listener import SirenEvent

        siren = SirenEvent(
            alert_id="test-1",
            title="בדקות הקרובות",
            zones=["רמת השרון"],
            matched_watch_zones=["רמת השרון"],
            is_active=True,
        )

        with patch("main.send_siren_alert", new_callable=AsyncMock) as mock_alert:
            await main._on_siren(siren)

        assert "רמת השרון" in main._pending_sirens
        # The alert should still fire (without trajectory match)
        mock_alert.assert_called_once()
        call_args = mock_alert.call_args
        assert call_args[0][1] is False  # trajectory_match=False

    @pytest.mark.asyncio
    async def test_pending_siren_cleared_after_expiry(self):
        """Pending sirens older than 60 seconds are pruned."""
        import main

        main._pending_sirens.clear()

        # Insert an expired siren (70 seconds ago)
        old_ts = datetime.now(timezone.utc).timestamp() - 70
        main._pending_sirens["רמת השרון"] = old_ts

        main._prune_pending_sirens()
        assert "רמת השרון" not in main._pending_sirens

    @pytest.mark.asyncio
    async def test_pending_siren_kept_when_fresh(self):
        """Pending sirens within 60 seconds are retained."""
        import main

        main._pending_sirens.clear()

        fresh_ts = datetime.now(timezone.utc).timestamp()
        main._pending_sirens["רמת השרון"] = fresh_ts

        main._prune_pending_sirens()
        assert "רמת השרון" in main._pending_sirens


class TestRetroactiveMatch:
    """Tests for _check_prediction_against_pending_sirens."""

    @pytest.mark.asyncio
    async def test_retroactive_match_fires_confirmed_alert(self):
        """An EKF prediction near a pending siren triggers OFFICIAL SIREN CONFIRMED."""
        import main

        main._pending_sirens.clear()

        # Plant a pending siren (just now)
        main._pending_sirens["רמת השרון"] = datetime.now(timezone.utc).timestamp()

        # Fake a prediction that lands 10km from Ramat HaSharon (32.1461, 34.8394)
        from unittest.mock import MagicMock

        prediction = MagicMock()
        prediction.impact_latitude = 32.19
        prediction.impact_longitude = 34.88
        prediction.terminal_velocity_m_s = 800.0
        prediction.seconds_until_impact = 120.0

        with patch("main.send_siren_alert", new_callable=AsyncMock) as mock_alert:
            await main._check_prediction_against_pending_sirens(
                prediction, "NORAD 99999"
            )

        mock_alert.assert_called_once()
        call_args = mock_alert.call_args
        assert call_args[0][1] is True  # trajectory_match=True
        assert "NORAD 99999" in call_args[0][2]  # match_summary

    @pytest.mark.asyncio
    async def test_no_retroactive_match_when_far_away(self):
        """An EKF prediction far from any pending siren does not trigger."""
        import main

        main._pending_sirens.clear()
        main._pending_sirens["רמת השרון"] = datetime.now(timezone.utc).timestamp()

        from unittest.mock import MagicMock

        prediction = MagicMock()
        prediction.impact_latitude = 29.0  # Eilat area — 300+ km away
        prediction.impact_longitude = 34.9
        prediction.terminal_velocity_m_s = 800.0
        prediction.seconds_until_impact = 120.0

        with patch("main.send_siren_alert", new_callable=AsyncMock) as mock_alert:
            await main._check_prediction_against_pending_sirens(
                prediction, "NORAD 99999"
            )

        mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_siren_consumed_after_match(self):
        """A matched pending siren is removed so it doesn't fire twice."""
        import main

        main._pending_sirens.clear()
        main._pending_sirens["רמת השרון"] = datetime.now(timezone.utc).timestamp()

        from unittest.mock import MagicMock

        prediction = MagicMock()
        prediction.impact_latitude = 32.19
        prediction.impact_longitude = 34.88
        prediction.terminal_velocity_m_s = 800.0
        prediction.seconds_until_impact = 120.0

        with patch("main.send_siren_alert", new_callable=AsyncMock):
            await main._check_prediction_against_pending_sirens(
                prediction, "NORAD 99999"
            )

        assert "רמת השרון" not in main._pending_sirens
