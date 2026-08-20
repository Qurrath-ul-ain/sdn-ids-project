#!/usr/bin/env python3

"""
Simple SDN-IDS mitigation module.

Reads detection results and creates block requests
for malicious source IP addresses.

Run from project root:

    python3 mitigation/response.py
"""

import json
import os
import time

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

RUNTIME_DIR = PROJECT_ROOT / "runtime"

DETECTION_FILE = (
    RUNTIME_DIR / "detection_results.json"
)

BLOCK_REQUEST_FILE = (
    RUNTIME_DIR / "block_requests.json"
)


def read_json(path):

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        return None


def write_json(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = Path(
        str(path) + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2
        )

    os.replace(
        temporary,
        path
    )


def create_block_requests():

    data = read_json(
        DETECTION_FILE
    )

    if not data:
        return

    results = data.get(
        "results",
        []
    )

    requests = []

    seen_ips = set()

    for result in results:

        prediction = result.get(
            "prediction"
        )

        source_ip = result.get(
            "source_ip"
        )

        if not source_ip:
            continue

        if prediction in (
            None,
            "Benign",
            "Unknown",
            "Error"
        ):
            continue

        if source_ip in seen_ips:
            continue

        seen_ips.add(
            source_ip
        )

        requests.append(
            {
                "source_ip": source_ip,
                "attack_type": prediction,
                "timestamp": time.time()
            }
        )

    if requests:

        write_json(
            BLOCK_REQUEST_FILE,
            requests
        )

        print(
            "[MITIGATION] Block request(s) created:"
        )

        for request in requests:

            print(
                f"  {request['source_ip']} "
                f"-> {request['attack_type']}"
            )


def main():

    RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "========================================"
    )

    print(
        "        SDN IDS MITIGATION"
    )

    print(
        "========================================"
    )

    print(
        "[MITIGATION] Waiting for detection results..."
    )

    last_timestamp = None

    while True:

        data = read_json(
            DETECTION_FILE
        )

        if data:

            timestamp = data.get(
                "timestamp"
            )

            if timestamp != last_timestamp:

                create_block_requests()

                last_timestamp = timestamp

        time.sleep(2)


if __name__ == "__main__":

    main()
