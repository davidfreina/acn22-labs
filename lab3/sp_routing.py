# This code is part of the Advanced Computer Networks (2020) course at Vrije
# Universiteit Amsterdam.

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

from ryu.base import app_manager
from ryu.controller import mac_to_port
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import packet
from ryu.lib.packet import ipv4
from ryu.lib.packet import arp
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types

from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
from ryu.app.wsgi import ControllerBase

import networkx as nx
import threading

import topo


class SPRouter(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SPRouter, self).__init__(*args, **kwargs)
        self.topo_net = topo.Fattree(4)
        self.num_switches = len(self.topo_net.switches)
        self.ip_to_port = {}
        self.controller_known_ips = []
        self.net = nx.DiGraph()
        self.topo_initialized = False
        self.logger.info("Please wait until the topology is initialized...")

    # Topology discovery
    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev):

        # Switches and links in the network
        switch_list = get_switch(self, None)
        switches = [switch.dp.id for switch in switch_list]
        self.net.add_nodes_from(switches)

        link_list = get_link(self, None)
        links = [(link.src.dpid, link.dst.dpid, {
                  'port': link.src.port_no}) for link in link_list]
        self.net.add_edges_from(links)
        links = [(link.dst.dpid, link.src.dpid, {
                  'port': link.dst.port_no}) for link in link_list]
        self.net.add_edges_from(links)

        if len(self.net.nodes()) == self.num_switches and not self.topo_initialized:
            self.logger.info("Finished initializing topology")
            self.topo_initialized = True

        # print(" \t" + "Current Links:")
        # for l in links:
        #     print (" \t\t" + str(l))

        # print(" \t" + "Current Switches:")
        # for s in switches:
        #     print (" \t\t" + str(s))

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install entry-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    # Add a flow entry to the flow-table

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Construct flow_mod message and send it
        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    def add_routing(self, ev, src_ip, dst_ip):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src

        if dst_ip in self.net:
            path = nx.shortest_path(self.net, src_ip, dst_ip)
            if dpid in path:
                next_hop = path[path.index(dpid) + 1]
                out_port = self.net[dpid][next_hop]['port']
            else:
                # print("dpid %s not in path from %s to %s", dpid, src_ip, dst_ip)
                return

        else:
            out_port = ofproto.OFPP_FLOOD

        # print(out_port)

        actions = [datapath.ofproto_parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=in_port, ipv4_dst=dst_ip, ipv4_src=src_ip)
            self.add_flow(datapath, 1, match, actions)

        out = datapath.ofproto_parser.OFPPacketOut(
            datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=in_port,
            actions=actions, data=msg.data)
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        # self.logger.info("\tpacket in: %s %s %s %s", dpid, src, dst, in_port)

        self.ip_to_port.setdefault(dpid, {})

        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            dst_ip = ip_pkt.dst
            src_ip = ip_pkt.src
            # self.logger.info("\tip packet in %s %s %s %s", dpid, src_ip, dst_ip, in_port)
            # print("\t\tip packet, adding routing")
            self.add_routing(ev, src_ip, dst_ip)

        elif eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocol(arp.arp)
            dst_ip = arp_pkt.dst_ip
            src_ip = arp_pkt.src_ip
            # self.logger.info("\tarp packet in %s %s %s %s", dpid, src_ip, dst_ip, in_port)

            self.ip_to_port.get(dpid)[src_ip] = in_port

            if src_ip not in self.controller_known_ips:
                self.controller_known_ips.append(src_ip)
                if src_ip not in self.net:
                    self.net.add_node(src_ip)
                    self.net.add_edge(dpid, src_ip, port=in_port)
                    self.net.add_edge(src_ip, dpid)

            # dst_ip has never been known therefore flood the ARP
            if dst_ip not in self.controller_known_ips:
                actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]

                out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER,
                                          in_port=in_port, actions=actions, data=msg.data)
                datapath.send_msg(out)
            else:
                # print("\t\tsuccessful arp response, adding routing")
                self.add_routing(ev, src_ip, dst_ip)
