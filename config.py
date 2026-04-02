"""
Centralized configuration loaded from environment variables.

All secrets (API keys, webhook URLs) are read from the environment so that
nothing sensitive is committed to source control.  A .env file is supported
for local development via pydantic-settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings - populated from env vars or .env file."""

    # -- NASA FIRMS --
    firms_api_key: str = ""
    firms_poll_interval_sec: int = 180
    firms_bounding_boxes: str = "-125,24,-66,50"

    # -- ADS-B Exchange --
    adsb_api_key: str = ""
    adsb_api_base_url: str = "https://adsbexchange.com/api/aircraft/v2"
    adsb_poll_interval_sec: int = 30
    adsb_watch_hex_codes: str = ""

    # -- Telegram Listener --
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_channels: str = ""

    # -- AWS --
    aws_region: str = "us-east-1"

    # -- Amazon Bedrock (LLM) --
    bedrock_model_id: str = "anthropic.claude-sonnet-4-20250514"

    # -- DynamoDB --
    dynamodb_table_name: str = "skyfall-events"
    dynamodb_endpoint_url: str = ""  # Override for local testing

    # -- Space-Track --
    spacetrack_identity: str = ""
    spacetrack_password: str = ""
    spacetrack_poll_interval_sec: int = 300  # 5 minutes

    # -- Alerting --
    slack_webhook_url: str = ""
    discord_webhook_url: str = ""

    # -- Logging --
    log_format: str = "json"  # "json" or "console"

    # -- Correlation tuning --
    correlation_window_sec: int = 300
    min_confidence_score: int = 6
    geohash_precision: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton - import this everywhere.
settings = Settings()
