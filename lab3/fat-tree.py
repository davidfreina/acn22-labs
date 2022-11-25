# This code is part of the Advanced Computer Networks (ACN) course at VU
# Amsterdam.

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

#!/usr/bin/env python3

# A dirty workaround to import topo.py from lab2

import os
import subprocess
import time

import mininet
import mininet.clean
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.log import lg, info
from mininet.link import TCLink
from mininet.node import Node, OVSKernelSwitch, RemoteController
from mininet.topo import Topo
from mininet.util import waitListening, custom

import topo


class FattreeNet(Topo):
    """
    Create a fat-tree network in Mininet
    """

    def __init__(self, ft_topo):

        Topo.__init__(self)
        edges = []
        edge_switches = []
        agg_switches = []
        core_switches = []

        for server in ft_topo.servers:
            for edge in server.edges:
                edge_sw = edge.lnode
                self.addHost("h{}".format(server.id.replace(".", "")), ip=server.id)
                if edge_sw not in edge_switches:
                    edge_switches.append(edge_sw)
                    self.addSwitch("s{}".format(edge_sw.id.replace(".", "")), ip=edge_sw.id)

                edges.append(edge)
                self.addLink("s{}".format(edge_sw.id.replace(".", "")), "h{}".format(server.id.replace(".", "")), bw=15, delay='5ms')

        for edge_sw in edge_switches:
            for edge in edge_sw.edges:
                if edge not in edges:
                    agg_sw = edge.lnode
                    if agg_sw not in agg_switches:
                        agg_switches.append(agg_sw)
                        self.addSwitch("s{}".format(agg_sw.id.replace(".", "")), ip=agg_sw.id)

                    edges.append(edge)
                    self.addLink("s{}".format(agg_sw.id.replace(".", "")),
                                 "s{}".format(edge_sw.id.replace(".", "")),
                                 bw=15, delay='5ms')

        for agg_sw in agg_switches:
            for edge in agg_sw.edges:
                if edge not in edges:
                    core_sw = edge.lnode
                    if core_sw not in core_switches:
                        core_switches.append(core_sw)
                        self.addSwitch("s{}".format(core_sw.id.replace(".", "")), ip=core_sw.id)

                    edges.append(edge)
                    self.addLink("s{}".format(core_sw.id.replace(".", "")),
                                 "s{}".format(agg_sw.id.replace(".", "")),
                                 bw=15, delay='5ms')


def make_mininet_instance(graph_topo):

    net_topo = FattreeNet(graph_topo)
    net = Mininet(topo=net_topo, controller=None, autoSetMacs=True)
    net.addController('c0', controller=RemoteController,
                      ip="127.0.0.1", port=6653)
    return net


def run(graph_topo):

    # Run the Mininet CLI with a given topology
    lg.setLogLevel('info')
    mininet.clean.cleanup()
    net = make_mininet_instance(graph_topo)

    info('*** Starting network ***\n')
    net.start()
    info('*** Running CLI***\n')
    CLI(net)
    info('*** Stopping network ***\n')
    net.stop()


def benchmark(graph_topo, warmup, avg):

    # Benchmarking different routings
    lg.setLogLevel('info')
    mininet.clean.cleanup()
    net = make_mininet_instance(graph_topo)

    info('*** Starting network ***\n')
    net.start()
    info('*** Running CLI, start RYU controller with specific routing in another terminal ***\n')
    net.waitConnected()

    # info('*** Running pingAll to flood the network ***\n')
    # start = time.time()
    # net.pingAll()
    # end = time.time()
    # warmup = end - start
    # time.sleep(5)

    # info('*** Running pingAll 10 times for benchmarking purposes ***\n')
    # for i in range(10):
    #     start = time.time()
    #     net.pingAll()
    #     end = time.time()
    #     tmp = end - start
    #     avg += tmp
    #     time.sleep(5)

    n1 = net.getNodeByName("h10002")

    n2 = net.getNodeByName("h10003")
    net.ping(hosts={n1, n2})
    #net.pingPair(tuple('h10002', 'h10003'))

    info('*** Stopping network ***\n')
    net.stop()

    return warmup, avg


ft_topo = topo.Fattree(4)
run(ft_topo)


warmup = 0
avg = 0

# #warmup, avg = benchmark(ft_topo, warmup, avg)
# print("Discovering topology took: \t{}".format(warmup))
# print("Average of 10 pingAll took: \t{}".format(avg / 10))
