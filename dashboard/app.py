from flask import Flask

app = Flask(__name__)


# ---------------------------------------------------------
# Standalone demo data
# This dashboard is intentionally independent.
# Detection and mitigation will be integrated later.
# ---------------------------------------------------------

total_packets = 500
normal_packets = 470
attack_packets = 30

detections = [
    {
        "source_ip": "10.0.0.4",
        "destination_ip": "10.0.0.2",
        "attack": "Brute Force",
        "confidence": "96%",
        "status": "Malicious",
    },
    {
        "source_ip": "10.0.0.5",
        "destination_ip": "10.0.0.3",
        "attack": "Botnet",
        "confidence": "94%",
        "status": "Malicious",
    },
    {
        "source_ip": "10.0.0.6",
        "destination_ip": "10.0.0.2",
        "attack": "Web Attack",
        "confidence": "93%",
        "status": "Malicious",
    },
    {
        "source_ip": "10.0.0.1",
        "destination_ip": "10.0.0.2",
        "attack": "Benign",
        "confidence": "95%",
        "status": "Normal",
    },
]

blocked_ips = [
    "10.0.0.4",
    "10.0.0.5",
    "10.0.0.6",
]


@app.route("/")
def dashboard():

    detection_rows = ""

    for item in detections:
        detection_rows += f"""
        <tr>
            <td>{item['source_ip']}</td>
            <td>{item['destination_ip']}</td>
            <td>{item['attack']}</td>
            <td>{item['confidence']}</td>
            <td>{item['status']}</td>
        </tr>
        """

    blocked_rows = ""

    for ip in blocked_ips:
        blocked_rows += f"<li>{ip}</li>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SDN IDS Dashboard</title>

        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                background: #f4f6f8;
            }}

            .header {{
                background: #1f2937;
                color: white;
                padding: 25px;
                text-align: center;
            }}

            .container {{
                width: 90%;
                margin: 30px auto;
            }}

            .cards {{
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }}

            .card {{
                background: white;
                padding: 20px;
                flex: 1;
                min-width: 180px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}

            .card h2 {{
                margin: 0 0 10px 0;
            }}

            .section {{
                background: white;
                margin-top: 25px;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
            }}

            th, td {{
                padding: 12px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}

            th {{
                background: #e5e7eb;
            }}

            ul {{
                line-height: 2;
            }}

            .normal {{
                color: green;
                font-weight: bold;
            }}

            .malicious {{
                color: red;
                font-weight: bold;
            }}

            .footer {{
                text-align: center;
                margin: 30px;
                color: #666;
            }}
        </style>
    </head>

    <body>

        <div class="header">
            <h1>SDN IDS Dashboard</h1>
            <p>Intrusion Detection and Mitigation Monitoring</p>
        </div>

        <div class="container">

            <div class="cards">

                <div class="card">
                    <h2>{total_packets}</h2>
                    <p>Total Packets</p>
                </div>

                <div class="card">
                    <h2>{normal_packets}</h2>
                    <p>Normal Traffic</p>
                </div>

                <div class="card">
                    <h2>{attack_packets}</h2>
                    <p>Attack Traffic</p>
                </div>

                <div class="card">
                    <h2>{len(blocked_ips)}</h2>
                    <p>Blocked IPs</p>
                </div>

            </div>


            <div class="section">

                <h2>Detection Results</h2>

                <table>

                    <tr>
                        <th>Source IP</th>
                        <th>Destination IP</th>
                        <th>Attack Type</th>
                        <th>Confidence</th>
                        <th>Status</th>
                    </tr>

                    {detection_rows}

                </table>

            </div>


            <div class="section">

                <h2>Mitigation Status</h2>

                <p>
                    Malicious IP addresses currently blocked by the
                    mitigation module:
                </p>

                <ul>
                    {blocked_rows}
                </ul>

            </div>

        </div>

        <div class="footer">
            SDN-Enabled Healthcare Network IDS
        </div>

    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
