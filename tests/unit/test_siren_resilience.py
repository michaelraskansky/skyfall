"""Tests for siren listener resilience: backoff and blindness alerting."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from ingestion.siren_listener import SirenEvent


class TestSirenBackoff:
    """Tests for jittered backoff on consecutive failures."""

    @pytest.mark.asyncio
    async def test_backoff_after_three_failures(self):
        """After 3 consecutive failures, sleep increases from 1s to ~5-7s."""
        from ingestion import siren_listener

        sleeps: list[float] = []

        async def mock_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) >= 5:
                raise asyncio.CancelledError  # Stop the loop

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        queue = asyncio.Queue()

        with (
            patch("ingestion.siren_listener.aiohttp.ClientSession", return_value=mock_session),
            patch("ingestion.siren_listener.asyncio.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await siren_listener.poll_sirens(queue)

        # First 2 sleeps should be 1s (normal), then after 3 failures -> backoff
        assert sleeps[0] == pytest.approx(1.0)
        assert sleeps[1] == pytest.approx(1.0)
        # Third consecutive failure triggers backoff: 5 + jitter(0, 2)
        assert 5.0 <= sleeps[2] <= 7.0

    @pytest.mark.asyncio
    async def test_counter_resets_on_success(self):
        """A successful response resets the failure counter back to 0."""
        from ingestion import siren_listener

        sleeps: list[float] = []
        responses = [403, 403, 200, 403, 403]

        async def mock_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) >= 5:
                raise asyncio.CancelledError

        mock_session = AsyncMock()

        statuses = iter(responses)

        def make_response():
            mock_response = AsyncMock()
            try:
                status = next(statuses)
            except StopIteration:
                raise asyncio.CancelledError
            mock_response.status = status
            mock_response.text = AsyncMock(return_value="[]")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            return mock_response

        mock_session.get = MagicMock(side_effect=lambda *a, **kw: make_response())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        queue = asyncio.Queue()

        with (
            patch("ingestion.siren_listener.aiohttp.ClientSession", return_value=mock_session),
            patch("ingestion.siren_listener.asyncio.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await siren_listener.poll_sirens(queue)

        # After the 200 response, failure counter resets.
        # None should be backoff since counter never hits 3 consecutively.
        assert all(s == pytest.approx(1.0) for s in sleeps[:4])


class TestBlindnessAlert:
    """Tests for the system warning when siren feed is blocked."""

    @pytest.mark.asyncio
    async def test_system_warning_on_three_failures(self):
        """After 3 consecutive failures, send_system_warning is called."""
        from ingestion import siren_listener

        call_count = 0

        async def mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 4:
                raise asyncio.CancelledError

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 502
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        queue = asyncio.Queue()

        with (
            patch("ingestion.siren_listener.aiohttp.ClientSession", return_value=mock_session),
            patch("ingestion.siren_listener.asyncio.sleep", side_effect=mock_sleep),
            patch("ingestion.siren_listener.send_system_warning", new_callable=AsyncMock) as mock_warn,
        ):
            with pytest.raises(asyncio.CancelledError):
                await siren_listener.poll_sirens(queue)

        mock_warn.assert_called_once()
        assert "Siren" in mock_warn.call_args[0][0]
