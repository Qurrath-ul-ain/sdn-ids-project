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

The exact feature list will be finalized before ML integration.

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
