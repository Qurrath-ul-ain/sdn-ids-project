#!/usr/bin/env python3

"""
Integrated Ryu controller for the SDN-IDS healthcare network.

Responsibilities:
1. Normal OpenFlow 1.3 learning-switch forwarding.
2. Periodically collect flow statistics.
3. Export flow statistics to runtime/flow_stats.json.
4. Read mitigation requests from runtime/block_requests.json.
5. Install a high-priority drop rule for blocked source IPs.

Run from the project root:

    ryu-manager controller/controller.py

The controller communicates with the IDS using simple JSON files.
"""

import json
import os
import time

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp, udp


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")

FLOW_STATS_FILE = os.path.join(
    RUNTIME_DIR, "flow_stats.json"
)

BLOCK_REQUEST_FILE = os.path.join(
    RUNTIME_DIR, "block_requests.json"
)

STATS_INTERVAL = 5


class HealthcareSwitch(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(HealthcareSwitch, self).__init__(*args, **kwargs)

        self.mac_to_port = {}

        self.datapaths = {}

        self.blocked_ips = set()

        os.makedirs(RUNTIME_DIR, exist_ok=True)

        self.monitor_thread = hub.spawn(self._monitor)

        self.logger.info(
            "Healthcare SDN-IDS controller started"
        )

    # ---------------------------------------------------------
    # SWITCH CONNECTION
    # ---------------------------------------------------------

    @set_ev_cls(
        ofp_event.EventOFPSwitchFeatures,
        CONFIG_DISPATCHER
    )
    def switch_features_handler(self, ev):

        datapath = ev.msg.datapath

        self.datapaths[datapath.id] = datapath

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()

        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]

        self.add_flow(
            datapath,
            0,
            match,
            actions
        )

        self.logger.info(
            "Switch connected: dpid=%s",
            datapath.id
        )

    @set_ev_cls(
        ofp_event.EventOFPStateChange,
        [MAIN_DISPATCHER, CONFIG_DISPATCHER]
    )
    def state_change_handler(self, ev):

        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:

            self.datapaths[datapath.id] = datapath

        elif ev.state == "DEAD_DISPATCHER":

            self.datapaths.pop(datapath.id, None)

    # ---------------------------------------------------------
    # ADD FLOW
    # ---------------------------------------------------------

    def add_flow(
        self,
        datapath,
        priority,
        match,
        actions,
        buffer_id=None
    ):

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        instructions = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]

        if buffer_id is not None:

            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=instructions
            )

        else:

            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=instructions
            )

        datapath.send_msg(mod)

    # ---------------------------------------------------------
    # PACKET IN
    # ---------------------------------------------------------

    @set_ev_cls(
        ofp_event.EventOFPPacketIn,
        MAIN_DISPATCHER
    )
    def packet_in_handler(self, ev):

        msg = ev.msg

        datapath = msg.datapath

        ofproto = datapath.ofproto

        parser = datapath.ofproto_parser

        dpid = datapath.id

        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)

        eth = pkt.get_protocol(
            ethernet.ethernet
        )

        if eth is None:
            return

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src

        self.mac_to_port.setdefault(
            dpid,
            {}
        )

        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:

            out_port = self.mac_to_port[dpid][dst]

        else:

            out_port = ofproto.OFPP_FLOOD

        actions = [
            parser.OFPActionOutput(out_port)
        ]

        if out_port != ofproto.OFPP_FLOOD:

            match = parser.OFPMatch(
                in_port=in_port,
                eth_dst=dst,
                eth_src=src
            )

            if msg.buffer_id != ofproto.OFP_NO_BUFFER:

                self.add_flow(
                    datapath,
                    1,
                    match,
                    actions,
                    msg.buffer_id
                )

                return

            self.add_flow(
                datapath,
                1,
                match,
                actions
            )

        data = None

        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )

        datapath.send_msg(out)

    # ---------------------------------------------------------
    # PERIODIC MONITOR
    # ---------------------------------------------------------

    def _monitor(self):

        while True:

            for datapath in list(
                self.datapaths.values()
            ):

                self._request_stats(datapath)

            self._process_block_requests()

            hub.sleep(STATS_INTERVAL)

    def _request_stats(self, datapath):

        parser = datapath.ofproto_parser

        request = parser.OFPFlowStatsRequest(
            datapath
        )

        datapath.send_msg(request)

    # ---------------------------------------------------------
    # FLOW STATISTICS
    # ---------------------------------------------------------

    @set_ev_cls(
        ofp_event.EventOFPFlowStatsReply,
        MAIN_DISPATCHER
    )
    def flow_stats_reply_handler(self, ev):

        datapath = ev.msg.datapath

        flows = []

        for stat in ev.msg.body:

            if stat.priority == 0:
                continue

            flow = self._convert_flow(
                stat
            )

            if flow is not None:

                flows.append(flow)

        result = {
            "timestamp": time.time(),
            "datapath_id": datapath.id,
            "flows": flows
        }

        self._write_json(
            FLOW_STATS_FILE,
            result
        )

        self.logger.info(
            "Exported %d flow(s) to %s",
            len(flows),
            FLOW_STATS_FILE
        )

    def _convert_flow(self, stat):

        match = stat.match

        ipv4_src = match.get("ipv4_src")
        ipv4_dst = match.get("ipv4_dst")

        if not ipv4_src or not ipv4_dst:
            return None

        protocol = match.get(
            "ip_proto",
            0
        )

        source_port = match.get(
            "tcp_src",
            match.get("udp_src", 0)
        )

        destination_port = match.get(
            "tcp_dst",
            match.get("udp_dst", 0)
        )

        duration_sec = getattr(
            stat,
            "duration_sec",
            0
        )

        duration_nsec = getattr(
            stat,
            "duration_nsec",
            0
        )

        duration_us = (
            int(duration_sec) * 1_000_000
            + int(duration_nsec) // 1_000
        )

        return {
            "source_ip": str(ipv4_src),
            "destination_ip": str(ipv4_dst),
            "source_port": int(source_port or 0),
            "destination_port": int(
                destination_port or 0
            ),
            "protocol": int(protocol or 0),
            "packet_count": int(
                getattr(stat, "packet_count", 0)
            ),
            "byte_count": int(
                getattr(stat, "byte_count", 0)
            ),
            "flow_duration_us": int(
                duration_us
            )
        }

    # ---------------------------------------------------------
    # MITIGATION
    # ---------------------------------------------------------

    def _process_block_requests(self):

        if not os.path.exists(
            BLOCK_REQUEST_FILE
        ):
            return

        try:

            with open(
                BLOCK_REQUEST_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                requests = json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):

            return

        if not isinstance(requests, list):
            return

        for request in requests:

            source_ip = request.get(
                "source_ip"
            )

            if not source_ip:
                continue

            if source_ip in self.blocked_ips:
                continue

            self._block_ip(
                source_ip
            )

            self.blocked_ips.add(
                source_ip
            )

        try:

            os.remove(
                BLOCK_REQUEST_FILE
            )

        except OSError:

            pass

    def _block_ip(self, source_ip):

        for datapath in list(
            self.datapaths.values()
        ):

            ofproto = datapath.ofproto

            parser = datapath.ofproto_parser

            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=source_ip
            )

            actions = []

            self.add_flow(
                datapath,
                100,
                match,
                actions
            )

            self.logger.warning(
                "BLOCKED source IP %s on switch %s",
                source_ip,
                datapath.id
            )

    # ---------------------------------------------------------
    # FILE HELPER
    # ---------------------------------------------------------

    @staticmethod
    def _write_json(path, data):

        temporary = path + ".tmp"

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
