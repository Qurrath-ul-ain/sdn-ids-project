from flask import Flask, render_template

app = Flask(__name__)

# Demo IDS data
summary = {
    "total_packets": 500,
    "normal_packets": 470,
    "attack_packets": 30,
    "blocked_ips": 3
}

detections = [
    {
        "source_ip": "10.0.0.4",
        "destination_ip": "10.0.0.2",
        "attack": "Brute Force",
        "confidence": 96,
        "status": "Malicious",
        "protocol": "TCP",
        "packets": 24
    },
    {
        "source_ip": "10.0.0.5",
        "destination_ip": "10.0.0.3",
        "attack": "Botnet",
        "confidence": 94,
        "status": "Malicious",
        "protocol": "TCP",
        "packets": 108
    },
    {
        "source_ip": "10.0.0.6",
        "destination_ip": "10.0.0.2",
        "attack": "Web Attack",
        "confidence": 93,
        "status": "Malicious",
        "protocol": "TCP",
        "packets": 35
    },
    {
        "source_ip": "10.0.0.1",
        "destination_ip": "10.0.0.2",
        "attack": "Benign",
        "confidence": 95,
        "status": "Normal",
        "protocol": "ICMP",
        "packets": 8
    }
]

blocked_ips = [
    {
        "ip": "10.0.0.4",
        "attack": "Brute Force",
        "status": "Blocked"
    },
    {
        "ip": "10.0.0.5",
        "attack": "Botnet",
        "status": "Blocked"
    },
    {
        "ip": "10.0.0.6",
        "attack": "Web Attack",
        "status": "Blocked"
    }
]


@app.route("/")
def dashboard():
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
