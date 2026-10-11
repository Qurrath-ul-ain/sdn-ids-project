import csv
import os
from datetime import datetime

OUTPUT = "data/raw/openflow_training_flows.csv"

FIELDS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "packet_count",
    "byte_count",
    "flow_duration_us",
    "label",
]

os.makedirs("data/raw", exist_ok=True)

if not os.path.exists(OUTPUT):
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

print(f"OpenFlow dataset file: {OUTPUT}")
print("Collector file created successfully.")
