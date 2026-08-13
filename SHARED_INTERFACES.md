# Shared Interfaces

## Flow Statistics

The controller will provide network flow information to the IDS.

Initial fields:

- source IP
- destination IP
- source port
- destination port
- protocol
- packet count
- byte count
- flow duration

## ML Input

The ML model will receive numerical network-flow features after preprocessing.

The finalized ML input is exactly these five numerical features, in this order:

1. `destination_port`
2. `protocol`
3. `packet_count`
4. `byte_count`
5. `flow_duration_us`

The ML interface rejects missing or unexpected feature keys. IP addresses and
other context fields are not part of the ML feature dictionary and should be
carried separately by the detection layer.

The final hybrid inference configuration uses RF weight `0.6` and DNN weight
`0.4`. Random Forest-only inference is also available because it remains the
strongest overall evaluated model.

## Detection Output

The IDS should produce:

- source IP
- destination IP
- prediction
- attack type
- confidence score

Example:

{
  "source_ip": "10.0.0.5",
  "destination_ip": "10.0.0.2",
  "prediction": "Malicious",
  "attack_type": "Brute Force",
  "confidence": 0.94
}


**QURRATH WORK**
## SDN Infrastructure — Stage 2–4 Status

### Stage 2 — Mininet Topology
- Healthcare SDN topology verified successfully.
- Hosts: h1, h2, h3, h4.
- Switch: s1.
- Ryu controller: 127.0.0.1:6653.
- OpenFlow version: OpenFlow 1.3.

### Stage 3 — Ryu Controller Verification
- Ryu successfully connected to switch s1.
- `pingall`: **0% dropped (12/12 received)**.
- `h1 -> h4`: **0% packet loss (3/3 received)**.
- OpenFlow traffic forwarding verified.

### Stage 4 — OpenFlow Verification
- Flow table verified using `ovs-ofctl`.
- Default `priority=0` controller rule confirmed.
- Dynamic `priority=1` forwarding rules confirmed.
- Flow packet/byte counters verified.
- Switch port packet/byte statistics verified.

### Status
- **Stage 2 — COMPLETE**
- **Stage 3 — COMPLETE**
- **Stage 4 — COMPLETE**
- **Stage 11 — PENDING**

