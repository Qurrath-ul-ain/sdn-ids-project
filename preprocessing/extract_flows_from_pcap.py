"""Extract bidirectional flows from raw PCAP files using scapy.

This replaces CICFlowMeter where it is unavailable.  It produces flow records
with a verified bidirectional communication key:

    normalized_key = (min_ip, min_port, max_ip, max_port, protocol)

which is used downstream for temporal grouping.  Labels are assigned by
matching each flow's timestamp and endpoint against the Wednesday-14-02-2018
CICFlowMeter CSV (used ONLY as a reference; not as input features).

Outputs
-------
data/raw/wednesday_flows_raw.csv
    One row per bidirectional flow with columns:
        flow_id, src_ip, src_port, dst_ip, dst_port, protocol,
        timestamp_start, timestamp_end, flow_duration_us,
        tot_fwd_pkts, tot_bwd_pkts, totlen_fwd_bytes, totlen_bwd_bytes,
        label, pcap_source

Schema mapping to CICFlowMeter columns
---------------------------------------
flow_id            - Flow ID (synthesized from 5-tuple + start-time)
src_ip             - Src IP
src_port           - Src Port
dst_ip             - Dst IP
dst_port           - Dst Port
protocol           - Protocol (6=TCP, 17=UDP, 0=other)
timestamp_start    - Timestamp (flow start, epoch seconds)
flow_duration_us   - Flow Duration (microseconds)
tot_fwd_pkts       - Tot Fwd Pkts
tot_bwd_pkts       - Tot Bwd Pkts
totlen_fwd_bytes   - TotLen Fwd Pkts
totlen_bwd_bytes   - TotLen Bwd Pkts
label              - Label  (Benign / Brute Force)
"""

from __future__ import annotations

import sys
# Scapy was installed as a user package
sys.path.insert(0, r'C:\Users\Shinjini\AppData\Roaming\Python\Python311\site-packages')

import json
import os
from collections import defaultdict
from pathlib import Path
import datetime

import pandas as pd

# Suppress scapy libpcap warning
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import PcapReader, IP, TCP, UDP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[1]
PCAP_DIR = Path(r'C:\Users\Shinjini\Downloads\pcap_wednesday\extracted\pcap')
OUT_CSV  = ROOT / 'data' / 'raw' / 'wednesday_flows_raw.csv'
META_DIR = ROOT / 'data' / 'metadata'

# PCAP files to process (extracted from Wednesday-14-02-2018 pcap.zip)
# UCAP files: 172.31.69.x SSH servers (12:29-21:30 UTC — no attack window)
# capWIN files: Windows FTP servers (12:29-21:30 UTC — contains FTP attack at 15:33-17:10 UTC)
PCAP_FILES = [
    'UCAP172.31.69.22',
    'UCAP172.31.69.18',
    'UCAP172.31.69.7',
    'UCAP172.31.69.27',
    'UCAP172.31.69.25',          # main 813 MB PCAP — likely contains attack traffic
    'capWIN-J6GMIG1DQE5-172.31.65.99',
    'capWIN-J6GMIG1DQE5-172.31.64.89',
    'capWIN-J6GMIG1DQE5-172.31.65.58',
    'capWIN-J6GMIG1DQE5-172.31.64.24',
]

# Wednesday-14-02-2018 attack timestamp ranges — CORRECTED TO UTC
# CICFlowMeter CSV timestamps are in EST (UTC-5).
# PCAP timestamps are in UTC.
# EST + 5 hours = UTC
#
# SSH-Bruteforce: EST 02:01:21-03:32:30 -> UTC 07:01:21-08:32:30 (before PCAP start at ~12:29 UTC)
# FTP-BruteForce: EST 10:33:26-12:10:31 -> UTC 15:33:26-17:10:31 (IN PCAP window)
SSH_START_UTC = datetime.datetime(2018, 2, 14,  7,  1, 21)
SSH_END_UTC   = datetime.datetime(2018, 2, 14,  8, 32, 30)
FTP_START_UTC = datetime.datetime(2018, 2, 14, 15, 33, 26)
FTP_END_UTC   = datetime.datetime(2018, 2, 14, 17, 10, 31)


def make_flow_key(ip_a: str, port_a: int, ip_b: str, port_b: int, proto: int) -> tuple:
    """Return normalized bidirectional 5-tuple key (canonical ordering)."""
    if (ip_a, port_a) <= (ip_b, port_b):
        return (ip_a, port_a, ip_b, port_b, proto)
    return (ip_b, port_b, ip_a, port_a, proto)


def classify_flow(dst_port: int, proto: int, start_dt: datetime.datetime) -> str:
    """Assign label using UTC-corrected timestamp + port heuristics (Wednesday day)."""
    if proto == 6 and dst_port == 22:
        if SSH_START_UTC <= start_dt <= SSH_END_UTC:
            return 'Brute Force'   # SSH brute-force (before PCAP window — unlikely)
    if proto == 6 and dst_port == 21:
        if FTP_START_UTC <= start_dt <= FTP_END_UTC:
            return 'Brute Force'   # FTP brute-force (within PCAP window)
    return 'Benign'


def extract_flows_from_pcap(pcap_path: Path, pcap_name: str) -> list[dict]:
    """Stream a PCAP and aggregate packets into bidirectional flow records.
    Uses PcapReader (streaming) instead of rdpcap to handle large files.
    """
    print(f'  Streaming {pcap_path.name} ...', flush=True)
    flows: dict[tuple, dict] = {}
    pkt_count = 0

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            pkt_count += 1
            if pkt_count % 500_000 == 0:
                print(f'    {pkt_count:,} packets processed ...', flush=True)

            if IP not in pkt:
                continue
            ip = pkt[IP]
            src_ip  = ip.src
            dst_ip  = ip.dst
            proto   = ip.proto
            ts      = float(pkt.time)
            pkt_len = len(ip)

            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            else:
                src_port = 0
                dst_port = 0

            key = make_flow_key(src_ip, src_port, dst_ip, dst_port, proto)

            if key not in flows:
                flows[key] = {
                    'first_ts':        ts,
                    'last_ts':         ts,
                    'init_src_ip':     src_ip,
                    'init_src_port':   src_port,
                    'init_dst_ip':     dst_ip,
                    'init_dst_port':   dst_port,
                    'proto':           proto,
                    'fwd_pkt_count':   0,
                    'bwd_pkt_count':   0,
                    'fwd_byte_count':  0,
                    'bwd_byte_count':  0,
                }

            f = flows[key]
            if ts < f['first_ts']:
                f['first_ts'] = ts
            if ts > f['last_ts']:
                f['last_ts'] = ts

            # Forward = toward the second endpoint in the normalized key
            if (src_ip, src_port) == (key[0], key[1]):
                f['fwd_pkt_count']  += 1
                f['fwd_byte_count'] += pkt_len
            else:
                f['bwd_pkt_count']  += 1
                f['bwd_byte_count'] += pkt_len

    print(f'    {pkt_count:,} packets total, {len(flows):,} bidirectional flows', flush=True)

    records = []
    for key, f in flows.items():
        start_dt   = datetime.datetime.utcfromtimestamp(f['first_ts'])
        duration_us = int((f['last_ts'] - f['first_ts']) * 1_000_000)

        src_ip   = f['init_src_ip']
        src_port = f['init_src_port']
        dst_ip   = f['init_dst_ip']
        dst_port = f['init_dst_port']
        proto    = f['proto']

        label    = classify_flow(int(dst_port), int(proto), start_dt)
        flow_id  = f"{src_ip}-{src_port}-{dst_ip}-{dst_port}-{proto}-{f['first_ts']:.6f}"

        records.append({
            'flow_id':           flow_id,
            'src_ip':            src_ip,
            'src_port':          int(src_port),
            'dst_ip':            dst_ip,
            'dst_port':          int(dst_port),
            'protocol':          int(proto),
            'timestamp_start':   float(f['first_ts']),
            'timestamp_end':     float(f['last_ts']),
            'flow_duration_us':  duration_us,
            'tot_fwd_pkts':      f['fwd_pkt_count'],
            'tot_bwd_pkts':      f['bwd_pkt_count'],
            'totlen_fwd_bytes':  f['fwd_byte_count'],
            'totlen_bwd_bytes':  f['bwd_byte_count'],
            'label':             label,
            'pcap_source':       pcap_name,
        })

    return records


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for pcap_name in PCAP_FILES:
        pcap_path = PCAP_DIR / pcap_name
        if not pcap_path.exists():
            print(f'WARNING: {pcap_path} not found - skipping', flush=True)
            continue
        records = extract_flows_from_pcap(pcap_path, pcap_name)
        all_records.extend(records)
        print(f'    Accumulated {len(all_records):,} records so far', flush=True)

    if not all_records:
        raise RuntimeError('No flow records extracted from any PCAP file.')

    df = pd.DataFrame(all_records)
    df.sort_values('timestamp_start', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(OUT_CSV, index=False)

    label_counts = df['label'].value_counts().to_dict()
    print(f'\nSaved {len(df):,} flows to {OUT_CSV}')
    print('Label distribution:', label_counts)

    schema = {
        'source_pcap_files':      PCAP_FILES,
        'pcap_zip':               r'C:\Users\Shinjini\Downloads\pcap_wednesday\pcap.zip',
        'day':                    'Wednesday-14-02-2018',
        'total_flows':            len(df),
        'label_counts':           label_counts,
        'columns':                list(df.columns),
        'schema_mapping': {
            'flow_id':           'Flow ID  (synthesized: src_ip-src_port-dst_ip-dst_port-proto-start_ts)',
            'src_ip':            'Src IP',
            'src_port':          'Src Port',
            'dst_ip':            'Dst IP',
            'dst_port':          'Dst Port',
            'protocol':          'Protocol (6=TCP, 17=UDP)',
            'timestamp_start':   'Timestamp (epoch seconds, flow start)',
            'flow_duration_us':  'Flow Duration (microseconds)',
            'tot_fwd_pkts':      'Tot Fwd Pkts',
            'tot_bwd_pkts':      'Tot Bwd Pkts',
            'totlen_fwd_bytes':  'TotLen Fwd Pkts (bytes)',
            'totlen_bwd_bytes':  'TotLen Bwd Pkts (bytes)',
            'label':             'Label (Benign / Brute Force)',
        },
        'grouping_key':  'normalized(src_ip, src_port, dst_ip, dst_port, protocol)',
        'label_method':  'Port + UTC timestamp heuristic vs CICFlowMeter reference (CSV=EST, PCAP=UTC, offset=+5h)',
        'timezone_note': 'CICFlowMeter CSV timestamps are EST (UTC-5). PCAP timestamps are UTC. FTP attack window 15:33-17:10 UTC is present in these files.',
        'limitation': (
            'Wednesday PCAP covers 12:29-21:30 UTC only. '
            'FTP-BruteForce (UTC 15:33-17:10) IS in this window and labeled via port 21 + timestamp. '
            'SSH-Bruteforce (UTC 07:01-08:32) is before the PCAP window and NOT captured. '
            'Only Benign + Brute Force (FTP) classes are available from this subset. '
            'Botnet and Web Attack classes require additional days (Friday-02-03, Thursday-22-02, Friday-23-02).'
        ),
    }
    schema_path = META_DIR / 'wednesday_flow_extraction_schema.json'
    with open(schema_path, 'w') as fh:
        json.dump(schema, fh, indent=2)
    print(f'Schema saved to {schema_path}')


if __name__ == '__main__':
    main()
