from flask import Flask, render_template
import os
import json

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RUNTIME_DIR = os.path.join(
    PROJECT_ROOT,
    "runtime"
)

DETECTIONS_FILE = os.path.join(
    RUNTIME_DIR,
    "detections.json"
)

BLOCK_FILE = os.path.join(
    RUNTIME_DIR,
    "blocked_ips.json"
)


def read_json(filename, default):

    try:

        if not os.path.exists(filename):
            return default

        with open(filename, "r") as f:
            return json.load(f)

    except (OSError, json.JSONDecodeError):

        return default


def get_dashboard_data():

    detections = read_json(
        DETECTIONS_FILE,
        []
    )

    block_requests = read_json(
        BLOCK_FILE,
        []
    )

    # Count packets, not detection records.
    normal_count = sum(
        int(item.get("packet_count", 0))
        for item in detections
        if item.get("status") == "Normal"
    )

    attack_count = sum(
        int(item.get("packet_count", 0))
        for item in detections
        if item.get("status") == "Malicious"
    )

    # Unique currently requested blocked IPs
    blocked_ips = block_requests

    blocked_details = []

    for ip in blocked_ips:

        attack = "Malicious"

        # Find the latest attack for this IP
        for item in reversed(detections):

            if item.get("source_ip") == ip:
                attack = item.get(
                    "attack",
                    "Malicious"
                )
                break

        blocked_details.append({
            "ip": ip,
            "attack": attack,
            "status": "Blocked"
        })

    summary = {
        "total_packets": sum(
            int(item.get("packet_count", 0))
            for item in detections
        ),
        "normal_packets": normal_count,
        "attack_packets": attack_count,
        "blocked_ips": len(blocked_ips)
    }

    return (
        summary,
        detections[-50:],
        blocked_details
    )


@app.route("/")
def dashboard():

    summary, detections, blocked_ips = (
        get_dashboard_data()
    )

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

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
