from flask import Flask

app = Flask(__name__)

# Sample data (later this can come from Detection and Mitigation)
total_packets = 500
normal_packets = 470
attack_packets = 30

blocked_ips = [
    "10.0.0.2",
    "10.0.0.5",
    "10.0.0.8"
]

@app.route("/")
def dashboard():

    html = f"""
    <html>
    <head>
        <title>SDN IDS Dashboard</title>
    </head>

    <body>

        <h1>SDN IDS Dashboard</h1>

        <hr>

        <h3>Total Packets : {total_packets}</h3>

        <h3>Normal Traffic : {normal_packets}</h3>

        <h3>Attack Traffic : {attack_packets}</h3>

        <h3>Blocked IPs</h3>

        <ul>
        {''.join([f'<li>{ip}</li>' for ip in blocked_ips])}
        </ul>

    </body>
    </html>
    """

    return html

if __name__ == "__main__":
    app.run(debug=True)
