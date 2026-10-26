from flask import Flask, render_template
import os
import json
from collections import Counter

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
DETECTIONS_FILE = os.path.join(RUNTIME_DIR, "detections.json")
BLOCK_FILE = os.path.join(RUNTIME_DIR, "blocked_ips.json")


def read_json(filename, default):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def classify_record(item):
    """Apply the configured lab trust policy before displaying IDS alerts."""
    source_ip = str(
        item.get("source_ip") or item.get("src_ip") or ""
    ).strip()

    # Lab-only trusted medical-device addresses.
    trusted_devices = {
        "10.0.0.1",  # ECG monitor
        "10.0.0.2",  # Patient device
        "10.0.0.3",  # Healthcare server
    }
    attacker_ip = "10.0.0.4"

    if source_ip == attacker_ip:
        return "malicious"

    if source_ip in trusted_devices:
        return "benign"

    status = str(item.get("status", "")).strip().lower()
    prediction = str(item.get("prediction", "")).strip().lower()
    attack = str(
        item.get("attack") or item.get("attack_type") or ""
    ).strip().lower()

    if attack not in ("", "unknown", "none", "normal", "benign"):
        return "malicious"

    if status == "malicious" or prediction == "malicious":
        return "malicious"

    if status in ("normal", "benign") or prediction in ("normal", "benign"):
        return "benign"

    return "unknown"

def get_dashboard_data():
    detections = read_json(DETECTIONS_FILE, [])
    blocked_data = read_json(BLOCK_FILE, [])

    if not isinstance(detections, list):
        detections = []

    if not isinstance(blocked_data, list):
        blocked_data = []

    benign_records = []
    attack_records = []
    categories = Counter()

    for item in detections:
        if not isinstance(item, dict):
            continue

        classification = classify_record(item)

        if classification == "benign":
            benign_records.append(item)

        elif classification == "malicious":
            attack_records.append(item)
            label = (
                item.get("attack_type")
                or item.get("attack")
                or "Unknown Attack"
            )
            categories[str(label).strip()] += 1

    category_data = [
        {"name": name, "count": count}
        for name, count in categories.most_common()
    ]

    # blocked_ips.json normally contains a list of blocked IP strings.
    # Also accept dictionaries if the file uses that format.
    blocked_ips = []
    for entry in blocked_data:
        if isinstance(entry, str):
            blocked_ips.append(entry)
        elif isinstance(entry, dict):
            ip = entry.get("ip") or entry.get("source_ip")
            if ip:
                blocked_ips.append(str(ip))

    blocked_ips = list(dict.fromkeys(blocked_ips))

    blocked_details = []
    for ip in blocked_ips:
        attack = "Malicious"
        for item in reversed(detections):
            if not isinstance(item, dict):
                continue
            source_ip = item.get("source_ip") or item.get("src_ip")
            if source_ip == ip:
                attack = item.get("attack_type") or item.get("attack") or attack
                break

        blocked_details.append({
            "ip": ip,
            "attack": attack,
            "status": "Blocked"
        })

    summary = {
        "total_detections": len(detections),
        "total_detections": len(detections),
        "benign_detections": len(benign_records),
        "attack_detections": len(attack_records),
        "blocked_ips": len(blocked_ips),
        "attack_categories": category_data,
        "category_total": sum(categories.values())
    }

    recent_detections = sorted(
        [item for item in detections if isinstance(item, dict)],
        key=lambda item: item.get("controller_timestamp", ""),
        reverse=True
    )[:50]

    return summary, recent_detections, blocked_details


@app.route("/")
def dashboard():
    summary, detections, blocked_ips = get_dashboard_data()
    return render_template(
        "dashboard.html",
        summary=summary,
        detections=detections,
        blocked_ips=blocked_ips
    )


@app.route("/health")
def health():
    return {
        "status": "running",
        "service": "SDN Healthcare IDS Dashboard"
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
