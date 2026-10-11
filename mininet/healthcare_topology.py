#!/usr/bin/env python3

from mininet.topo import Topo


class HealthcareTopo(Topo):
    def build(self):
        patient_monitor = self.addHost(
            'patient_monitor',
            ip='10.0.0.1/24'
        )

        infusion_pump = self.addHost(
            'infusion_pump',
            ip='10.0.0.2/24'
        )

        mri_scanner = self.addHost(
            'mri_scanner',
            ip='10.0.0.3/24'
        )

        attacker = self.addHost(
            'attacker_workstation',
            ip='10.0.0.4/24'
        )

        switch = self.addSwitch('s1')

        self.addLink(patient_monitor, switch, intfName1='pm-eth0')
        self.addLink(infusion_pump, switch, intfName1='ip-eth0')
        self.addLink(mri_scanner, switch, intfName1='mri-eth0')
        self.addLink(attacker, switch, intfName1='atk-eth0')


topos = {
    'healthcare': lambda: HealthcareTopo()
}
