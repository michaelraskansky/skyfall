"""Shared test fixtures for the Skyfall test suite."""

import os
from datetime import datetime, timezone
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from models import EventSource, RawEvent


def _patch_aiobotocore_for_moto():
    """Make moto's sync AWSResponse.content awaitable for aiobotocore.

    moto intercepts HTTP requests and returns botocore AWSResponse objects
    with a synchronous ``.content`` property (returns ``bytes``).
    aiobotocore, however, awaits ``.content`` because the real aiohttp
    response is async.  This shim makes the ``bytes`` result awaitable so
    the two libraries work together in tests.
    """
    from botocore.awsrequest import AWSResponse

    _original_content_fget = AWSResponse.content.fget

    class _AwaitableBytes(bytes):
        """bytes subclass whose instances can be awaited (returns self)."""

        def __await__(self):
            async def _identity():
                return bytes(self)

            return _identity().__await__()

    @property  # type: ignore[misc]
    def _patched_content(self):
        raw = _original_content_fget(self)
        if isinstance(raw, bytes):
            return _AwaitableBytes(raw)
        return raw

    AWSResponse.content = _patched_content


# Apply the patch at import time so every async DynamoDB test benefits.
_patch_aiobotocore_for_moto()


@pytest.fixture
def aws_credentials():
    """Mock AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    yield
    for key in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ.pop(key, None)


@pytest.fixture
def dynamodb_table(aws_credentials):
    """Create a moto DynamoDB table matching the skyfall-events schema."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        client.create_table(
            TableName="skyfall-events",
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        client.update_time_to_live(
            TableName="skyfall-events",
            TimeToLiveSpecification={
                "Enabled": True,
                "AttributeName": "expires_at",
            },
        )
        yield client


def make_raw_event(
    source: EventSource = EventSource.FIRMS,
    lat: float = 29.76,
    lon: float = -95.37,
    description: str = "Test event",
    event_id: str | None = None,
) -> RawEvent:
    """Factory for creating RawEvent instances in tests."""
    kwargs = {
        "source": source,
        "latitude": lat,
        "longitude": lon,
        "description": description,
        "timestamp": datetime.now(timezone.utc),
        "raw_payload": {"text": description},
    }
    if event_id:
        kwargs["event_id"] = event_id
    return RawEvent(**kwargs)
