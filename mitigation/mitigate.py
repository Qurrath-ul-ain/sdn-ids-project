#!/usr/bin/env python3
"""Create persistent mitigation requests for the SDN controller."""

import ipaddress
import json
import os
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
BLOCK_REQUEST_FILE = os.path.join(RUNTIME_DIR, "block_requests.json")


def _validate_ip(ip: str) -> str:
    try:
        address = ipaddress.ip_address(ip)
    except ValueError as exc:
        raise ValueError(f"Invalid IP address: {ip}") from exc
    if address.version != 4:
        raise ValueError("Only IPv4 addresses are supported.")
    return str(address)


def _load_requests() -> List[dict]:
    if not os.path.exists(BLOCK_REQUEST_FILE):
        return []
    try:
        with open(BLOCK_REQUEST_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _write_requests(requests: List[dict]) -> None:
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    temporary_file = BLOCK_REQUEST_FILE + ".tmp"
    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(requests, file, indent=2)
    os.replace(temporary_file, BLOCK_REQUEST_FILE)


def block_ip(ip: str) -> bool:
    source_ip = _validate_ip(ip)
    requests = _load_requests()

    if any(
        item.get("source_ip") == source_ip and item.get("type") != "flow"
        for item in requests if isinstance(item, dict)
    ):
        print(f"IP {source_ip} is already pending mitigation.")
        return False

    requests.append({"source_ip": source_ip})
    _write_requests(requests)
    print(f"Full IP block request created for {source_ip}")
    return True


def block_flow(source_ip: str, destination_ip: str,
               destination_port: int, protocol: int = 6,
               attack_type: str = "Botnet") -> bool:
    source_ip = _validate_ip(source_ip)
    destination_ip = _validate_ip(destination_ip)
    destination_port = int(destination_port)
    protocol = int(protocol)

    if protocol not in (6, 17):
        raise ValueError("Only TCP and UDP are supported.")
    if not 1 <= destination_port <= 65535:
        raise ValueError("Destination port must be between 1 and 65535.")

    request = {
        "type": "flow",
        "source_ip": source_ip,
        "destination_ip": destination_ip,
        "protocol": protocol,
        "destination_port": destination_port,
        "attack_type": attack_type
    }

    requests = _load_requests()
    identity = (
        source_ip, destination_ip, protocol, destination_port
    )

    for item in requests:
        if not isinstance(item, dict) or item.get("type") != "flow":
            continue
        existing = (
            item.get("source_ip"),
            item.get("destination_ip"),
            item.get("protocol"),
            item.get("destination_port")
        )
        if existing == identity:
            print(f"Targeted flow restriction is already pending: {identity}")
            return False

    requests.append(request)
    _write_requests(requests)
    print(
        f"Targeted flow restriction requested: "
        f"{source_ip} -> {destination_ip}, protocol={protocol}, "
        f"destination_port={destination_port}"
    )
    return True


def mitigation(prediction: str, ip: str, destination_ip=None,
               destination_port=None, protocol=6) -> bool:
    prediction_normalized = prediction.strip().lower()

    if prediction_normalized not in {
        "attack", "brute force", "botnet", "web attack"
    }:
        print("Normal traffic. No mitigation required.")
        return False

    print(f"Attack detected from {ip}")

    if (
        prediction_normalized == "botnet"
        and destination_ip is not None
        and destination_port is not None
    ):
        return block_flow(
            ip,
            destination_ip,
            destination_port,
            protocol,
            "Botnet"
        )

    return block_ip(ip)
