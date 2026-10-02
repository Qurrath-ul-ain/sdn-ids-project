#!/usr/bin/env python3
"""
Healthcare SDN IDS topology with an external attacker.

External network:
    External Attacker  192.168.100.10

Gateway:
    r1
    External side:    192.168.100.1/24
    Healthcare side:  10.0.0.254/24

Healthcare network:
    ECG Monitor       10.0.0.1
    Patient Monitor   10.0.0.2
    Healthcare Server 10.0.0.3
    Infusion Pump     10.0.0.4

All endpoints are virtual Mininet hosts.
No physical medical devices are required.
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Node
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink


def build_topology():
    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    info('*** Adding SDN controller\n')
    c0 = net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6653
    )

    info('*** Adding healthcare switch\n')
    s1 = net.addSwitch(
        's1',
        protocols='OpenFlow13'
    )

    info('*** Adding external switch\n')
    s2 = net.addSwitch(
        's2',
        protocols='OpenFlow13'
    )

    info('*** Adding virtual healthcare devices\n')
    ecg = net.addHost(
        'ecg',
        ip='10.0.0.1/24',
        defaultRoute='via 10.0.0.254'
    )

    patient = net.addHost(
        'patient',
        ip='10.0.0.2/24',
        defaultRoute='via 10.0.0.254'
    )

    server = net.addHost(
        'server',
        ip='10.0.0.3/24',
        defaultRoute='via 10.0.0.254'
    )

    infusion = net.addHost(
        'infusion',
        ip='10.0.0.4/24',
        defaultRoute='via 10.0.0.254'
    )

    info('*** Adding external attacker\n')
    attacker = net.addHost(
        'attacker',
        ip='192.168.100.10/24',
        defaultRoute='via 192.168.100.1'
    )

    info('*** Adding virtual healthcare gateway\n')
    r1 = net.addHost(
        'r1',
        ip='10.0.0.254/24'
    )

    info('*** Creating healthcare links\n')
    net.addLink(ecg, s1)
    net.addLink(patient, s1)
    net.addLink(server, s1)
    net.addLink(infusion, s1)
    net.addLink(r1, s1, params1={'ip': '10.0.0.254/24'})

    info('*** Creating external network\n')
    net.addLink(attacker, s2)
    net.addLink(r1, s2, params1={'ip': '192.168.100.1/24'})

    info('*** Starting network\n')
    net.build()

    c0.start()
    s1.start([c0])
    s2.start([c0])

    info('*** Enabling IP forwarding on gateway\n')
    r1.cmd('sysctl -w net.ipv4.ip_forward=1')

    info('*** Healthcare SDN IDS network started\n')
    CLI(net)

    info('*** Stopping network\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    build_topology()
