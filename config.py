"""
Centralized configuration loaded from environment variables.

All secrets (API keys, webhook URLs) are read from the environment so that
nothing sensitive is committed to source control.  A .env file is supported
for local development via pydantic-settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings – populated from env vars or .env file."""

    # ── NASA FIRMS ────────────────────────────────────────────────────────
    firms_api_key: str = ""
    firms_poll_interval_sec: int = 180  # 3 minutes
    # Comma-separated bounding boxes: "west,south,east,north;..."
    firms_bounding_boxes: str = "-125,24,-66,50"  # CONUS default

    # ── ADS-B Exchange ────────────────────────────────────────────────────
    adsb_api_key: str = ""
    adsb_api_base_url: str = "https://adsbexchange.com/api/aircraft/v2"
    adsb_poll_interval_sec: int = 30
    # Hex codes of high-altitude survey / government aircraft (comma-sep)
    adsb_watch_hex_codes: str = ""

    # ── Telegram Listener ─────────────────────────────────────────────────
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_channels: str = ""  # comma-separated channel usernames

    # ── LLM (OpenAI-compatible or Gemini) ─────────────────────────────────
    llm_provider: str = "openai"  # "openai" | "gemini"
    openai_api_key: str = ""
    gemini_api_key: str = ""
    llm_model: str = "gpt-4o"

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Alerting ──────────────────────────────────────────────────────────
    slack_webhook_url: str = ""
    discord_webhook_url: str = ""

    # ── Correlation tuning ────────────────────────────────────────────────
    correlation_window_sec: int = 300  # 5-minute correlation window
    min_confidence_score: int = 6  # LLM confidence threshold (1-10)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton – import this everywhere.
settings = Settings()
