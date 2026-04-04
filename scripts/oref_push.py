"""
Pikud HaOref → Skyfall Push Proxy
===================================

Polls the Pikud HaOref siren API every second from an Israeli IP
and pushes new alerts to the Skyfall /api/v1/siren endpoint.

Run on a machine with an Israeli IP:
    nohup python3 oref_push.py > oref_push.log 2>&1 &

Zero dependencies — Python stdlib only.
"""

import json
import time
import urllib.request

OREF_URL = "https://www.oref.org.il/warningMessages/alert/Alerts.json"
SKYFALL_URL = "https://wjube6a2u7.execute-api.eu-central-1.amazonaws.com/api/v1/siren"
API_KEY = "jLKwijEbMCXtHtNh_zTZiFtnCImlq5GSLNoVOO1qS38"

OREF_HEADERS = {
    "Referer": "https://www.oref.org.il/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json",
}

seen_ids = set()

print("Oref push proxy starting...", flush=True)

while True:
    try:
        req = urllib.request.Request(OREF_URL, headers=OREF_HEADERS)
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode()

        # Oref API returns UTF-8 BOM — strip it
        text = text.lstrip("\ufeff")

        if not text or text.strip() in ("", "[]"):
            time.sleep(1)
            continue

        alerts = json.loads(text)
        if isinstance(alerts, dict):
            alerts = [alerts]

        for alert in alerts:
            alert_id = str(alert.get("id", ""))
            if not alert_id or alert_id in seen_ids:
                continue
            seen_ids.add(alert_id)

            push_req = urllib.request.Request(
                SKYFALL_URL,
                data=json.dumps(alert).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": API_KEY,
                },
                method="POST",
            )
            with urllib.request.urlopen(push_req, timeout=5) as push_resp:
                result = push_resp.read().decode()
            print(
                f"Pushed: {alert.get('cat', '?')} | "
                f"{alert.get('title', '')[:60]} | {result}",
                flush=True,
            )

    except Exception as e:
        print(f"Error: {e}", flush=True)

    if len(seen_ids) > 10000:
        seen_ids.clear()

    time.sleep(1)
