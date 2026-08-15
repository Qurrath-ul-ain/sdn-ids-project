#!/usr/bin/env python3

import argparse
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen


def run_command(command):
    """Run a system command and display its output."""
    print(f"\n[+] Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=False)
    except FileNotFoundError:
        print(f"[!] Command not found: {command[0]}")


def normal_ping(target):
    """Generate normal ICMP traffic."""
    print(f"\n[+] Generating normal ping traffic to {target}")
    run_command(["ping", "-c", "5", target])


def normal_iperf(target, port):
    """Generate normal TCP traffic using iperf3."""
    print(f"\n[+] Generating normal iperf3 traffic to {target}:{port}")

    run_command([
        "iperf3",
        "-c",
        target,
        "-p",
        str(port),
        "-t",
        "10"
    ])


def normal_http(url):
    """Generate normal HTTP GET traffic."""
    print(f"\n[+] Generating normal HTTP traffic to {url}")

    for i in range(5):
        try:
            with urlopen(url, timeout=3) as response:
                print(f"HTTP request {i + 1}: status={response.status}")
        except Exception as error:
            print(f"HTTP request {i + 1}: {error}")

        time.sleep(1)


def tcp_connection_attempt(target, port):
    """Create one TCP connection attempt."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect((target, port))
        sock.close()
    except (ConnectionRefusedError, TimeoutError, OSError):
        pass


def brute_force_simulation(target, port, attempts):
    """
    Simulate brute-force-like connection behavior.

    This does not perform password attacks.
    It only creates repeated TCP connection attempts
    against the local SDN test environment.
    """
    print(
        f"\n[+] Simulating brute-force traffic "
        f"against {target}:{port}"
    )

    for i in range(attempts):
        tcp_connection_attempt(target, port)

    print(f"[+] Completed {attempts} connection attempts")


def botnet_simulation(target, port, workers, requests):
    """
    Simulate botnet-like distributed traffic by creating
    concurrent TCP connections from multiple workers.
    """
    print(
        f"\n[+] Simulating botnet-style traffic "
        f"to {target}:{port}"
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(tcp_connection_attempt, target, port)
            for _ in range(requests)
        ]

        for future in futures:
            future.result()

    print(f"[+] Completed {requests} concurrent connection attempts")


def web_attack_simulation(base_url):
    """
    Generate abnormal HTTP request patterns for IDS testing.

    These are harmless test requests and do not contain
    destructive SQL injection or exploit payloads.
    """
    test_paths = [
        "/admin",
        "/login",
        "/robots.txt",
        "/unknown-page",
        "/search?q=test",
        "/search?q=aaaaaaaaaaaaaaaaaaaaaaaa"
    ]

    print(f"\n[+] Simulating web-attack traffic to {base_url}")

    for path in test_paths:
        url = base_url.rstrip("/") + path

        try:
            with urlopen(url, timeout=3) as response:
                print(f"{url} -> HTTP {response.status}")
        except Exception as error:
            print(f"{url} -> {error}")

        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare SDN traffic generator"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "ping",
            "iperf",
            "http",
            "bruteforce",
            "botnet",
            "webattack"
        ],
        required=True
    )

    parser.add_argument(
        "--target",
        default="10.0.0.2",
        help="Target IP address"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=80,
        help="Target TCP port"
    )

    parser.add_argument(
        "--url",
        default="http://10.0.0.2",
        help="HTTP target URL"
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=50,
        help="Number of connection attempts"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent workers"
    )

    args = parser.parse_args()

    if args.mode == "ping":
        normal_ping(args.target)

    elif args.mode == "iperf":
        normal_iperf(args.target, args.port)

    elif args.mode == "http":
        normal_http(args.url)

    elif args.mode == "bruteforce":
        brute_force_simulation(
            args.target,
            args.port,
            args.attempts
        )

    elif args.mode == "botnet":
        botnet_simulation(
            args.target,
            args.port,
            args.workers,
            args.attempts
        )

    elif args.mode == "webattack":
        web_attack_simulation(args.url)


if __name__ == "__main__":
    main()
