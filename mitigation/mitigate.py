#!/usr/bin/env python3

"""
SDN IDS mitigation module.

When the detection system identifies malicious traffic,
this module writes a block request for the SDN controller.

The controller reads:
    runtime/block_requests.json

and installs the actual OpenFlow drop rule.
"""

import ipaddress
import json
import os
import sys
from typing import List


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RUNTIME_DIR = os.path.join(
    PROJECT_ROOT,
    "runtime"
)

BLOCK_REQUEST_FILE = os.path.join(
    RUNTIME_DIR,
    "block_requests.json"
)


def _validate_ip(ip: str) -> str:
    """Validate and normalize an IPv4 address."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError as exc:
        raise ValueError(f"Invalid IP address: {ip}") from exc

    if address.version != 4:
        raise ValueError("Only IPv4 addresses are supported.")

    return str(address)


def _load_requests() -> List[dict]:
    """Load existing mitigation requests."""
    if not os.path.exists(BLOCK_REQUEST_FILE):
        return []

    try:
        with open(
            BLOCK_REQUEST_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    return data


def _write_requests(requests: List[dict]) -> None:
    """Atomically write mitigation requests."""
    os.makedirs(RUNTIME_DIR, exist_ok=True)

    temporary_file = BLOCK_REQUEST_FILE + ".tmp"

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            requests,
            file,
            indent=2
        )

    os.replace(
        temporary_file,
        BLOCK_REQUEST_FILE
    )


def block_ip(ip: str) -> bool:
    """
    Request the SDN controller to block an IP.

    Returns True when a new block request is created.
    Returns False when the IP is already pending.
    """

    source_ip = _validate_ip(ip)

    requests = _load_requests()

    for request in requests:
        if request.get("source_ip") == source_ip:
            print(
                f"IP {source_ip} is already pending mitigation."
            )
            return False

    requests.append(
        {
            "source_ip": source_ip
        }
    )

    _write_requests(requests)

    print(
        f"Mitigation request created for {source_ip}"
    )

    return True


def mitigation(prediction: str, ip: str) -> bool:
    """
    Trigger mitigation when malicious traffic is detected.

    Expected malicious prediction:
        Attack

    Normal traffic:
        No mitigation.
    """

    if prediction.strip().lower() != "attack":
        print(
            "Normal traffic. No mitigation required."
        )
        return False

    print(
        f"Attack detected from {ip}"
    )

    return block_ip(ip)


def main() -> None:
    """Simple command-line test."""

    if len(sys.argv) != 3:
        print(
            "Usage: python3 mitigation/mitigate.py "
            "<prediction> <source_ip>"
        )
        sys.exit(1)

    prediction = sys.argv[1]
    source_ip = sys.argv[2]

    mitigation(
        prediction,
        source_ip
    )


if __name__ == "__main__":
    main()
