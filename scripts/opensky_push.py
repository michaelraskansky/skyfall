"""
OpenSky Network → Skyfall ADS-B Push Proxy
============================================

Polls the OpenSky public API every 10 seconds and pushes aircraft state
batches to the Skyfall /api/v1/adsb endpoint.

Run locally (OpenSky blocks cloud IPs):
    python3 scripts/opensky_push.py

Zero dependencies — Python stdlib only.
"""

import json
import time
import urllib.request

SKYFALL_URL = "https://wjube6a2u7.execute-api.eu-central-1.amazonaws.com/api/v1/adsb"
API_KEY = "jLKwijEbMCXtHtNh_zTZiFtnCImlq5GSLNoVOO1qS38"
POLL_INTERVAL = 30  # seconds (OpenSky anonymous rate limit: 10 req/min)

# Bounding box: lamin, lomin, lamax, lomax (entire Middle East)
BBOX = "12,25,42,63"

OPENSKY_URL = "https://opensky-network.org/api/states/all"

# OpenSky response field indices
_ICAO24 = 0
_CALLSIGN = 1
_ORIGIN_COUNTRY = 2
_LONGITUDE = 5
_LATITUDE = 6
_BARO_ALT = 7
_ON_GROUND = 8
_VELOCITY = 9
_TRUE_TRACK = 10
_SQUAWK = 14


def build_opensky_url():
    parts = [p.strip() for p in BBOX.split(",")]
    if len(parts) == 4:
        return (
            f"{OPENSKY_URL}"
            f"?lamin={parts[0]}&lomin={parts[1]}"
            f"&lamax={parts[2]}&lomax={parts[3]}"
        )
    return OPENSKY_URL


def parse_states(data):
    """Convert OpenSky positional arrays to keyed dicts for Skyfall."""
    aircraft = []
    for ac in data.get("states") or []:
        if len(ac) < 17:
            continue
        hex_code = ac[_ICAO24]
        if not hex_code:
            continue
        aircraft.append({
            "hex": hex_code,
            "callsign": (ac[_CALLSIGN] or "").strip(),
            "lat": ac[_LATITUDE],
            "lon": ac[_LONGITUDE],
            "track": ac[_TRUE_TRACK],
            "alt_m": ac[_BARO_ALT],
            "velocity_m_s": ac[_VELOCITY],
            "origin_country": ac[_ORIGIN_COUNTRY] or "",
            "on_ground": bool(ac[_ON_GROUND]),
            "squawk": ac[_SQUAWK],
        })
    return aircraft


def push_to_skyfall(aircraft):
    """POST aircraft batch to Skyfall endpoint."""
    payload = json.dumps({"aircraft": aircraft}).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    req = urllib.request.Request(
        SKYFALL_URL,
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def main():
    url = build_opensky_url()
    print(f"OpenSky push proxy starting", flush=True)
    print(f"  OpenSky URL: {url}", flush=True)
    print(f"  Skyfall URL: {SKYFALL_URL}", flush=True)
    print(f"  Interval:    {POLL_INTERVAL}s", flush=True)
    print(f"  Bbox:        {BBOX}", flush=True)
    print(flush=True)

    while True:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            aircraft = parse_states(data)

            if aircraft:
                result = push_to_skyfall(aircraft)
                anomalies = result.get("anomalies", 0)
                status = f"  → {result.get('total', '?')} accepted, {anomalies} anomalies"
                if anomalies > 0:
                    status = f"  ⚠ {result.get('total', '?')} accepted, {anomalies} ANOMALIES"
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Polled {len(aircraft)} aircraft{status}",
                    flush=True,
                )
            else:
                print(
                    f"[{time.strftime('%H:%M:%S')}] No aircraft in bounding box",
                    flush=True,
                )

        except urllib.error.HTTPError as e:
            print(f"[{time.strftime('%H:%M:%S')}] HTTP {e.code}: {e.reason}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}", flush=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
