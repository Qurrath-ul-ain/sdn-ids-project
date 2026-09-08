#!/usr/bin/env python3
"""
topology.py
Stage 2 - Basic healthcare SDN topology for the IDS project.

Layout:
    h1 (normal host)   \
    h2 (normal host)    >--- s1 (OVS switch, OpenFlow13) --- Ryu controller (127.0.0.1:6653)
    h3 (medical server)/
    h4 (attacker host) /

Run with:
    sudo python3 topology.py

This script assumes a Ryu controller is (or will be) listening on
127.0.0.1:6653. In Stage 2 we are only testing the topology itself,
so Mininet's default reference controller behavior does not apply -
we explicitly point at a RemoteController. If no controller is running
yet, the switch will connect but hosts won't be able to ping each other
(that's expected and fine for now - Stage 2 just verifies the topology
builds and hosts/links come up correctly).
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink


def build_topology():
    net = Mininet(controller=RemoteController, switch=OVSSwitch, link=TCLink, autoSetMacs=True)

    info('*** Adding controller\n')
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    info('*** Adding switch\n')
    s1 = net.addSwitch('s1', protocols='OpenFlow13')

    info('*** Adding hosts\n')
    h1 = net.addHost('h1', ip='10.0.0.1/24')  # normal host
    h2 = net.addHost('h2', ip='10.0.0.2/24')  # normal host
    h3 = net.addHost('h3', ip='10.0.0.3/24')  # medical server
    h4 = net.addHost('h4', ip='10.0.0.4/24')  # attacker host

    info('*** Creating links\n')
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s1)
    net.addLink(h4, s1)

    info('*** Starting network\n')
    net.build()
    c0.start()
    s1.start([c0])

    info('*** Running CLI\n')
    CLI(net)

    info('*** Stopping network\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    build_topology()
