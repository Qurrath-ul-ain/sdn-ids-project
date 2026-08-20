"""
Mitigation Module
Blocks malicious flows detected by the IDS.
Supports:
- Brute Force
- Botnet
- Web Attack
"""

from datetime import datetime

class MitigationEngine:

    def __init__(self):
        self.blocked_flows = []

    def mitigate(self, detection):

        attack = detection["attack"]

        if attack == "Normal":
            return {
                "status": "ALLOWED",
                "action": "No mitigation required"
            }

        source = detection.get("source_ip", "Unknown")
        destination = detection.get("destination_ip", "Unknown")

        action = self.block_flow(source, destination)

        return {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attack": attack,
            "source": source,
            "destination": destination,
            "status": "BLOCKED",
            "action": action
        }

    def block_flow(self, source, destination):

        flow = {
            "source": source,
            "destination": destination
        }

        self.blocked_flows.append(flow)

        print(f"[MITIGATION] Blocking flow {source} --> {destination}")

        # In a real SDN environment, send an OpenFlow rule here.
        # Example:
        # datapath.send_msg(flow_mod)

        return "Flow rule installed successfully"

    def get_blocked_flows(self):
        return self.blocked_flows


if __name__ == "__main__":

    engine = MitigationEngine()

    sample_detection = {
        "attack": "Botnet",
        "source_ip": "10.0.0.5",
        "destination_ip": "10.0.0.12"
    }

    result = engine.mitigate(sample_detection)

    print("\n=== Mitigation Result ===")
    for key, value in result.items():
        print(f"{key}: {value}")
