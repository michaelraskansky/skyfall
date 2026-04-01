# Real-Time Aerospace Debris & Industrial Anomaly Tracker

A Python-based pipeline that detects unregistered space debris re-entries and severe industrial accidents by correlating multiple public data streams in real time.

## Architecture

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐
│  FIRMS   │  │  ADS-B   │  │ Telegram │  │  Emergency  │
│  Poller  │  │  Poller  │  │ Listener │  │  Webhook    │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘
     └─────────────┴──────┬──────┴───────────────┘
                          ▼
                ┌───────────────────┐
                │  Raw Event Queue  │
                └────────┬──────────┘
                         ▼
                ┌───────────────────┐
                │   Triage Loop     │
                │  LLM + Correlate  │
                └────────┬──────────┘
                         ▼
                ┌───────────────────┐
                │   Alert Queue     │
                └────────┬──────────┘
                         ▼
                ┌───────────────────┐
                │  Slack / Discord  │
                └───────────────────┘
```

### Layer 1 – Data Ingestion
| Module | Source | Polling Interval |
|---|---|---|
| `ingestion/firms_poller.py` | NASA FIRMS VIIRS 375m | 3 min |
| `ingestion/adsb_poller.py` | ADS-B Exchange | 30 sec |
| `ingestion/emergency_webhook.py` | FastAPI POST endpoint | Push |
| `ingestion/social_listener.py` | Telegram / generic scraper | Real-time |

### Layer 2 – Processing & Triage
| Module | Role |
|---|---|
| `processing/llm_parser.py` | LLM-based text → structured JSON |
| `processing/correlation_engine.py` | Redis-backed geo-temporal correlation |

### Layer 3 – Output & Alerting
| Module | Role |
|---|---|
| `output/formatter.py` | Build clean alert payloads |
| `output/alerter.py` | Push to Slack / Discord webhooks |

## Quick Start

```bash
# 1. Copy and fill in secrets
cp .env.example .env

# 2. Run with Docker Compose
docker compose up --build

# 3. Test the emergency webhook
curl -X POST http://localhost:8000/api/v1/emergency \
  -H "Content-Type: application/json" \
  -d '{"source_system":"test","event_type":"explosion","latitude":29.76,"longitude":-95.37,"description":"Large fireball seen downtown"}'
```

## AWS Deployment

- **Worker**: EC2 instance or ECS/Fargate task running the Docker image.
- **Redis**: Amazon ElastiCache (Redis 7+). Set `REDIS_URL` to the ElastiCache endpoint.
- **Webhook ingress**: ALB or API Gateway in front of port 8000.

## Directory Structure

```
debris-tracker/
├── main.py                          # Orchestrator entry point
├── config.py                        # Pydantic settings (env vars)
├── models.py                        # Shared data models
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── ingestion/
│   ├── firms_poller.py              # NASA FIRMS thermal anomaly poller
│   ├── adsb_poller.py               # ADS-B airspace disruption poller
│   ├── emergency_webhook.py         # FastAPI emergency endpoint
│   └── social_listener.py           # Telegram / generic social scraper
├── processing/
│   ├── llm_parser.py                # LLM triage (OpenAI / Gemini)
│   └── correlation_engine.py        # Redis-backed correlation engine
├── output/
│   ├── formatter.py                 # Alert payload builder
│   └── alerter.py                   # Slack / Discord webhook delivery
└── tests/
```
