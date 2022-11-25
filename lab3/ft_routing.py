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
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ipv4
from ryu.lib.packet import arp
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types

from ryu.topology import event
from ryu.topology.api import get_switch, get_link

import topo

class FTRouter(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(FTRouter, self).__init__(*args, **kwargs)
        self.topo_net = topo.Fattree(4)
        self.routing_tables = {}
        self.create_ft_routing(4)


    # Topology discovery
    @set_ev_cls(event.EventSwitchEnter)
    def get_topology_data(self, ev):
        # Switches and links in the network
        switches = get_switch(self, None)
        links = get_link(self, None)

    def create_ft_routing(self, k):
        for pod in range(k):
            for switch in range(k // 2, k):
                # edge switch
                self.routing_tables.setdefault("10.{}.{}.1".format(pod, switch - k // 2), {})
                # agg switch
                self.routing_tables.setdefault("10.{}.{}.1".format(pod, switch), {})

                for subnet in range(k // 2):
                    # edge switch
                    self.routing_tables["10.{}.{}.1".format(pod, switch - k // 2)].setdefault("10.{}.{}.{}/32".format(pod, switch - k // 2, subnet + 2), subnet + 1)
                    # agg switch
                    self.routing_tables["10.{}.{}.1".format(pod, switch)].setdefault("10.{}.{}.0/24".format(pod, subnet), subnet + 1)

                # edge switch
                self.routing_tables["10.{}.{}.1".format(pod, switch - k // 2)].setdefault("0.0.0.0/0", {})
                # agg switch
                self.routing_tables["10.{}.{}.1".format(pod, switch)].setdefault("0.0.0.0/0", {})

                for host in range(2, k // 2 + 2):
                    port = ((host - 2 + switch) % (k // 2)) + (k // 2)
                    # edge switch
                    self.routing_tables["10.{}.{}.1".format(pod, switch - k // 2)]["0.0.0.0/0"].setdefault("0.0.0.{}/8".format(host), port + 1)
                    # agg switch
                    self.routing_tables["10.{}.{}.1".format(pod, switch)]["0.0.0.0/0"].setdefault("0.0.0.{}/8".format(host), port + 1)

        for j in range(1, k // 2 + 1):
            for i in range(1, k // 2 + 1):
                self.routing_tables.setdefault("10.{}.{}.{}".format(k, j, i), {})
                for destPod in range(k):
                    self.routing_tables["10.{}.{}.{}".format(k, j, i)].setdefault("10.{}.0.0/16".format(destPod), destPod + 1)

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
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    def find_out_port(self, dpid, dst_ip):
        current_switch = [sw for sw in self.topo_net.switches if int(sw.id.replace(".", "")) == dpid]


        if current_switch:
            current_switch = current_switch[0]

            for prefix in self.routing_tables[current_switch.id]:
                port = self.routing_tables[current_switch.id][prefix]
                prefix, subnet = prefix.split("/")
                prefix_length = int(subnet) // 8

                if subnet == "0":
                    # handle suffix
                    for suffix in port:
                        suffix_port = port[suffix]
                        suffix, subnet = suffix.split("/")
                        suffix_length = int(subnet) // 8
                        # self.logger.info("suffix: %s, subnet: %s, suffix_length: %s, suffix_port: %s", suffix, subnet, suffix_length, suffix_port)


                        if dst_ip.split(".")[::-1][:suffix_length] == suffix.split(".")[::-1][:suffix_length]:
                            # self.logger.info("out_port: %s", suffix_port)
                            return suffix_port


                if dst_ip.split(".")[:prefix_length] == prefix.split(".")[:prefix_length]:
                    # self.logger.info("out_port: %s", port)
                    return port

        return None

    def send_packet(self, ev, out_port, is_ipv4 = False):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        actions = [parser.OFPActionOutput(out_port)]

        if is_ipv4:
            match = parser.OFPMatch(in_port=in_port, ipv4_dst=is_ipv4)
            self.add_flow(datapath, 1, match, actions)

        out = parser.OFPPacketOut(datapath=datapath,
                                  in_port=in_port,
                                  actions=actions,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  data=msg.data)

        datapath.send_msg(out)


    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        dpid = ev.msg.datapath.id
        data = ev.msg.data

        # Get packet to identify destination and source
        pkt = packet.Packet(data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            dst_ip = ip_pkt.dst
            out_port = self.find_out_port(dpid, dst_ip)

            if out_port is not None:
                self.send_packet(ev, out_port, dst_ip)
            # else:
            #     self.logger.info("no out_port found: %s, %s", dpid, dst_ip)

        elif eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocol(arp.arp)
            dst_ip = arp_pkt.dst_ip
            out_port = self.find_out_port(dpid, dst_ip)

            if out_port is not None:
                self.send_packet(ev, out_port)
            # else:
            #     self.logger.info("arp no out_port found: %s, %s", dpid, dst_ip)
        else:
            return

