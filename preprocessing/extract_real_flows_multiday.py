"""Extract flows from multiple days' PCAP files using a high-speed binary PCAP parser.
Applies correct timezone-aware UTC datetime conversions and robust port checks.
"""
import struct
import socket
import datetime
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\Shinjini\sdn-ids-project")
RAW_DIR = ROOT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RAW_DIR / 'real_flows_multiday.csv'

# PCAP files configuration
DOWNLOADED_DIR = Path(r"C:\Users\Shinjini\Downloads\extracted_pcap_days")
pcap_configs = [
    # 1. Wednesday (Brute Force)
    {
        'day': 'Wednesday-14-02-2018',
        'path': Path(r"C:\Users\Shinjini\Downloads\pcap_wednesday\extracted\pcap\UCAP172.31.69.25"),
        'attack_type': 'Brute Force',
        'attack_port': 21,
        'start_utc': datetime.datetime(2018, 2, 14, 14, 33, 26, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 2, 14, 16, 10, 31, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': True
    },
    # 2. Friday (Botnet)
    {
        'day': 'Friday-02-03-2018',
        'path': DOWNLOADED_DIR / 'Friday-02-03-2018' / 'UCAP172.31.69.7',
        'attack_type': 'Botnet',
        'attack_port': 8080,
        'start_utc': datetime.datetime(2018, 3, 2, 14, 17, 7, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 3, 2, 20, 0, 0, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': False
    },
    {
        'day': 'Friday-02-03-2018',
        'path': DOWNLOADED_DIR / 'Friday-02-03-2018' / 'UCAP172.31.69.15',
        'attack_type': 'Botnet',
        'attack_port': 8080,
        'start_utc': datetime.datetime(2018, 3, 2, 14, 17, 7, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 3, 2, 20, 0, 0, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': False
    },
    {
        'day': 'Friday-02-03-2018',
        'path': DOWNLOADED_DIR / 'Friday-02-03-2018' / 'UCAP172.31.69.28',
        'attack_type': 'Botnet',
        'attack_port': 8080,
        'start_utc': datetime.datetime(2018, 3, 2, 14, 17, 7, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 3, 2, 20, 0, 0, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': False
    },
    # 3. Thursday (Web Attack)
    {
        'day': 'Thursday-22-02-2018',
        'path': DOWNLOADED_DIR / 'Thursday-22-02-2018' / 'UCAP172.31.69.28',
        'attack_type': 'Web Attack',
        'attack_port': 80,
        'start_utc': datetime.datetime(2018, 2, 22, 14, 13, 44, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 2, 22, 16, 0, 0, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': False
    },
    {
        'day': 'Thursday-22-02-2018',
        'path': DOWNLOADED_DIR / 'Thursday-22-02-2018' / 'UCAP172.31.69.15',
        'attack_type': 'Web Attack',
        'attack_port': 80,
        'start_utc': datetime.datetime(2018, 2, 22, 14, 13, 44, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 2, 22, 16, 0, 0, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': False
    },
    {
        'day': 'Thursday-22-02-2018',
        'path': DOWNLOADED_DIR / 'Thursday-22-02-2018' / 'UCAP172.31.69.21',
        'attack_type': 'Web Attack',
        'attack_port': 80,
        'start_utc': datetime.datetime(2018, 2, 22, 14, 13, 44, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 2, 22, 16, 0, 0, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': False
    },
    # 4. Friday (Web Attack)
    {
        'day': 'Friday-23-02-2018',
        'path': DOWNLOADED_DIR / 'Friday-23-02-2018' / 'UCAP172.31.69.28',
        'attack_type': 'Web Attack',
        'attack_port': 80,
        'start_utc': datetime.datetime(2018, 2, 23, 14, 2, 37, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 2, 23, 16, 30, 0, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': False
    },
    {
        'day': 'Friday-23-02-2018',
        'path': DOWNLOADED_DIR / 'Friday-23-02-2018' / 'UCAP172.31.69.27',
        'attack_type': 'Web Attack',
        'attack_port': 80,
        'start_utc': datetime.datetime(2018, 2, 23, 14, 2, 37, tzinfo=datetime.timezone.utc).timestamp(),
        'end_utc':   datetime.datetime(2018, 2, 23, 16, 30, 0, tzinfo=datetime.timezone.utc).timestamp(),
        'sampling': False
    }
]

def make_flow_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: int) -> tuple:
    """Normalized flow key for bidirectional grouping."""
    if (src_ip, src_port) < (dst_ip, dst_port):
        return (src_ip, src_port, dst_ip, dst_port, proto)
    else:
        return (dst_ip, dst_port, src_ip, src_port, proto)

def parse_pcap_fast(pcap_path):
    """High-speed binary parser for PCAP files."""
    with open(pcap_path, 'rb') as f:
        global_header = f.read(24)
        if len(global_header) < 24:
            return
        magic = struct.unpack('<I', global_header[:4])[0]
        swapped = False
        nano = False
        if magic == 0xd4c3b2a1:
            swapped = True
        elif magic == 0xa1b23c4d:
            nano = True
        elif magic == 0x4d3cb2a1:
            swapped = True
            nano = True
        elif magic != 0xa1b2c3d4:
            raise ValueError(f"Unsupported magic number: {hex(magic)}")
            
        fmt_char = '>' if swapped else '<'
        header_fmt = fmt_char + 'IIII'
        
        while True:
            header_data = f.read(16)
            if len(header_data) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(header_fmt, header_data)
            pkt_data = f.read(incl_len)
            if len(pkt_data) < incl_len:
                break
            
            if len(pkt_data) < 34:
                continue
                
            ethertype = struct.unpack('>H', pkt_data[12:14])[0]
            if ethertype != 0x0800:
                continue
                
            version_ihl = pkt_data[14]
            ihl = version_ihl & 0x0f
            ip_len = ihl * 4
            proto = pkt_data[14 + 9]
            
            if proto not in (6, 17):
                continue
                
            src_ip = socket.inet_ntoa(pkt_data[14+12:14+16])
            dst_ip = socket.inet_ntoa(pkt_data[14+16:14+20])
            
            tcp_udp_start = 14 + ip_len
            if len(pkt_data) < tcp_udp_start + 4:
                continue
                
            sport, dport = struct.unpack('>HH', pkt_data[tcp_udp_start : tcp_udp_start+4])
            ts = ts_sec + (ts_usec / 1e9 if nano else ts_usec / 1e6)
            
            yield ts, src_ip, sport, dst_ip, dport, proto, len(pkt_data) - 14

all_records = []

for cfg in pcap_configs:
    pcap_path = cfg['path']
    if not pcap_path.exists():
        print(f"Skipping missing file: {pcap_path}")
        continue
    
    print(f"Processing {pcap_path.name} ({cfg['day']}) ...", flush=True)
    flows = {}
    pkt_count = 0
    
    for ts, src_ip, sport, dst_ip, dport, proto, pkt_len in parse_pcap_fast(pcap_path):
        pkt_count += 1
        if pkt_count % 1_000_000 == 0:
            print(f"  {pkt_count:,} packets parsed...", flush=True)
            
        is_attack_pkt = (dport == cfg['attack_port'] or sport == cfg['attack_port']) and (cfg['start_utc'] <= ts <= cfg['end_utc'])
        
        # Sampling logic for Wednesday UCAP25
        if cfg['sampling'] and not is_attack_pkt:
            if pkt_count % 100 != 0:
                continue
                
        key = make_flow_key(src_ip, sport, dst_ip, dport, proto)
        
        if key not in flows:
            flows[key] = {
                'first_ts': ts,
                'last_ts': ts,
                'src_ip': src_ip,
                'src_port': sport,
                'dst_ip': dst_ip,
                'dst_port': dport,
                'protocol': proto,
                'fwd_pkts': 0,
                'bwd_pkts': 0,
                'fwd_bytes': 0,
                'bwd_bytes': 0,
            }
            
        f = flows[key]
        if ts < f['first_ts']: f['first_ts'] = ts
        if ts > f['last_ts']: f['last_ts'] = ts
        
        if (src_ip, sport) == (key[0], key[1]):
            f['fwd_pkts'] += 1
            f['fwd_bytes'] += pkt_len
        else:
            f['bwd_pkts'] += 1
            f['bwd_bytes'] += pkt_len

    print(f"  Total packets: {pkt_count:,}, unique flows: {len(flows):,}", flush=True)
    
    # Classify flows
    for key, f in flows.items():
        duration = max(1.0, (f['last_ts'] - f['first_ts']) * 1e6)
        
        # Attack port check on either source or destination (bidirectional flow)
        is_attack_flow = (f['dst_port'] == cfg['attack_port'] or f['src_port'] == cfg['attack_port']) and (cfg['start_utc'] <= f['first_ts'] <= cfg['end_utc'])
        label = cfg['attack_type'] if is_attack_flow else 'Benign'
        
        all_records.append({
            'flow_id': f"{f['src_ip']}-{f['src_port']}-{f['dst_ip']}-{f['dst_port']}-{f['protocol']}",
            'src_ip': f['src_ip'],
            'src_port': f['src_port'],
            'dst_ip': f['dst_ip'],
            'dst_port': f['dst_port'],
            'protocol': f['protocol'],
            'timestamp': f['first_ts'],
            'flow_duration_us': duration,
            'packet_count': f['fwd_pkts'] + f['bwd_pkts'],
            'byte_count': f['fwd_bytes'] + f['bwd_bytes'],
            'label': label,
            'day': cfg['day']
        })

df = pd.DataFrame(all_records)
print(f"\nTotal extracted flows: {len(df):,}")
print("Label counts:")
print(df['label'].value_counts())

df.to_csv(OUT_CSV, index=False)
print(f"Saved to {OUT_CSV}")
