# Productionize Skyfall: Testing, Service Migration & Deployment

## Overview

Migrate Skyfall from a prototype (Redis + OpenAI/Gemini, no tests, docker-compose only) to a production-ready system on AWS (DynamoDB + Bedrock, full test suite, ECS/Fargate deployment with CloudFormation).

**Approach:** Bottom-up — migrate services first, then test, then infra.

## 1. DynamoDB Migration

Replace Redis-backed correlation engine with DynamoDB.

### Table Design

| Key | Attribute | Purpose |
|---|---|---|
| PK | `geohash#<4-char>` | Geohash prefix (~20km grid cell) — groups nearby events |
| SK | `<ISO-timestamp>#<event_id>` | Enables time-range queries within a cell |
| Attributes | `source`, `lat`, `lon`, `raw_payload`, `description` | Event data |
| TTL | `expires_at` (epoch seconds) | Auto-cleanup after correlation window + buffer |

### Correlation Logic

1. **Ingest:** Compute geohash from lat/lon, write event to DynamoDB.
2. **Correlate:** Query the event's geohash cell plus its 8 neighbors. Filter by time window (`correlation_window_sec`, default 300s). Check for 2+ distinct sources.
3. **Cleanup:** DynamoDB TTL handles expiry automatically.

### Geohash Helper

Pure Python implementation (~30 lines). No external library needed. Provides:
- `encode(lat, lon, precision) -> str`
- `neighbors(geohash) -> list[str]` (8 surrounding cells)

4-char geohash gives ~20km cells. With neighbor expansion, effective search area is ~60km on a side — comparable to the current 50km Redis GEOSEARCH radius.

### Dependencies

- Remove: `redis[hiredis]`
- Add: `aioboto3` (async DynamoDB client, brings `boto3` transitively)

## 2. Bedrock LLM Migration

Replace OpenAI and Gemini SDK calls with Amazon Bedrock (Claude).

### Changes

- Single Bedrock path via `bedrock-runtime` `invoke_model`
- Default model: `anthropic.claude-sonnet-4-20250514` (configurable)
- System prompt and JSON output schema remain identical
- Use `aioboto3` for native async calls (consistent with DynamoDB client)
- Auth via IAM — no API keys needed for LLM

### Config Changes

| Remove | Add |
|---|---|
| `openai_api_key` | `bedrock_model_id` (default: `anthropic.claude-sonnet-4-20250514`) |
| `gemini_api_key` | `aws_region` (default: `us-east-1`) |
| `llm_provider` | — |
| `llm_model` | — |

### Dependencies

- Remove: `openai`, `google-generativeai`

## 3. Testing Strategy

Three layers. No mocking of application logic. AWS service fakes (`moto`, `respx`) at service boundaries.

### Unit Tests (`tests/unit/`)

Pure functions, no I/O:

- **`trajectory/physics.py`** — coordinate transforms, atmosphere model against known reference values
- **`trajectory/predictor.py`** — EKF with synthetic observations, verify impact prediction within tolerance
- **`output/formatter.py`** — feed CorrelatedEvent, assert output structure
- **`models.py`** — validation edge cases (confidence bounds, enum values)
- **Geohash helper** — known lat/lon to expected geohash, neighbor computation

### Integration Tests (`tests/integration/`)

Real code, faked AWS services:

- **`processing/correlation_engine.py`** — `moto` DynamoDB. Test ingest, time-window queries, neighbor-cell expansion, multi-source correlation to CRITICAL elevation, TTL attribute presence.
- **`processing/llm_parser.py`** — `moto` Bedrock. Seed canned Claude response, verify JSON parsing into LLMParsedEvent, test malformed LLM output handling.
- **`ingestion/emergency_webhook.py`** — FastAPI `TestClient`. POST valid/invalid payloads, verify queue receives events.
- **`output/alerter.py`** — `respx` (httpx mock). Capture outbound webhook calls, verify payload shape and retry behavior.

### End-to-End Test (`tests/e2e/`)

- Wire up full pipeline (queues, triage, correlation, alerter) with moto DynamoDB and moto Bedrock
- Inject synthetic RawEvents from 2+ sources in the same geo-cell and time window
- Assert a CorrelatedEvent with CRITICAL severity is produced and an alert is dispatched

### Test Dependencies (dev)

- `moto[dynamodb,bedrock]`
- `respx`

## 4. Operational Hardening

### Structured Logging

- Switch from `logging.basicConfig` to structlog JSON renderer
- Context fields on every log line: `event_id`, `source`, `correlation_id` where applicable
- Log levels: INFO (normal flow), WARNING (LLM parse failures), ERROR (retryable failures), CRITICAL (task crashes)

### Health Checks

- Implement `/health` endpoint on the FastAPI app
- Verify DynamoDB table reachable, Bedrock model accessible
- Used by ECS task health and ALB target health

### Graceful Shutdown

- Add SIGTERM handler (ECS sends this on deploy/scale-down)
- Drain raw queue before exiting, flush pending alerts
- Close DynamoDB/Bedrock clients cleanly

### Retry & Resilience

- Extend `tenacity` retries to Bedrock calls (transient throttling) and DynamoDB (ProvisionedThroughputExceeded)
- Circuit breaker for external pollers (FIRMS, ADS-B) — exponential backoff when APIs are down

### Config Validation

- Startup checks: DynamoDB table exists, Bedrock model is accessible
- Fail fast on boot if critical config is missing

## 5. CloudFormation & Deployment

### Resources

| Resource | Type | Notes |
|---|---|---|
| DynamoDB table | `AWS::DynamoDB::Table` | On-demand billing, TTL on `expires_at` |
| ECS Cluster | `AWS::ECS::Cluster` | Fargate capacity provider |
| ECS Task Definition | `AWS::ECS::TaskDefinition` | Fargate, 0.5 vCPU / 1GB |
| ECS Service | `AWS::ECS::Service` | Desired count: 1 |
| ALB + Target Group | `AWS::ElasticLoadBalancingV2` | Routes to port 8000 |
| ECR Repository | `AWS::ECR::Repository` | Docker image store |
| IAM Task Role | `AWS::IAM::Role` | `bedrock:InvokeModel`, `dynamodb:*` on table |
| IAM Execution Role | `AWS::IAM::Role` | ECR pull, CloudWatch Logs |
| CloudWatch Log Group | `AWS::Logs::LogGroup` | 30-day retention |
| Secrets Manager | `AWS::SecretsManager::Secret` | FIRMS key, ADS-B key, Telegram creds, webhook URLs |

### Template Parameters

- `VpcId` — existing VPC
- `SubnetIds` — existing subnets
- `ImageUri` — ECR image URI

### Dockerfile Update

- Switch from `requirements.txt` to `uv pip install` from `pyproject.toml`
- Base image stays Python 3.12 slim

### Not Included

CI/CD pipeline, custom domain, WAF, auto-scaling policies — to be added later.

## 6. Dependency Changes

### Remove

- `openai` — replaced by Bedrock
- `google-generativeai` — replaced by Bedrock
- `redis[hiredis]` — replaced by DynamoDB
- `telethon` — no Telegram account; keep code, make import optional

### Add (main)

- `aioboto3` — async DynamoDB + Bedrock

### Add (dev)

- `moto[dynamodb,bedrock]` — AWS service fakes
- `respx` — httpx mock for alerter tests

### Keep

`aiohttp`, `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `numpy`, `scipy`, `geopy`, `python-dotenv`, `structlog`, `tenacity`, `pytest`, `pytest-asyncio`

## Implementation Order

1. DynamoDB migration (correlation engine rewrite + geohash helper)
2. Bedrock LLM migration (llm_parser rewrite)
3. Config/dependency updates (pyproject.toml, settings, Dockerfile) — prerequisite for all test steps
4. Unit tests
5. Integration tests
6. End-to-end test
7. Operational hardening (logging, health, shutdown, retries)
8. CloudFormation template
