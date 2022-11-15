# This code is part of the Advanced Computer Networks course at Vrije
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

import sys
import random
import queue
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import ConnectionPatch
from itertools import count

# Class for an edge in the graph


class Edge:
    def __init__(self):
        self.lnode = None
        self.rnode = None

    def __repr__(self):
        return "\n\t\tEdge({}, {})".format(self.lnode.id, self.rnode.id)

    def __str__(self):
        return "\n\t\tEdge({}, {})".format(self.lnode.id, self.rnode.id)

    def remove(self):
        self.lnode.edges.remove(self)
        self.rnode.edges.remove(self)
        self.lnode = None
        self.rnode = None

# Class for a node in the graph


class Node:
    def __init__(self, id, type):
        self.edges = []
        self.id = id
        self.type = type

    def __repr__(self):
        # return "Node({})".format(self.id, self.type, self.edges)
        return "\tNode({}, {}, {})\n".format(self.id, self.type, self.edges)

    def __str__(self):
        return "\tNode({}, {}, {})\n".format(self.id, self.type, self.edges)

    # Add an edge connected to another node
    def add_edge(self, node):
        edge = Edge()
        edge.lnode = self
        edge.rnode = node
        self.edges.append(edge)
        node.edges.append(edge)
        return edge

    # Remove an edge from the node
    def remove_edge(self, edge):
        self.edges.remove(edge)

    # Decide if another node is a neighbor
    def is_neighbor(self, node):
        for edge in self.edges:
            if edge.lnode == node or edge.rnode == node:
                return True
        return False


class Jellyfish:

    def __init__(self, num_servers, num_switches, num_ports):
        self.servers = []
        self.switches = []
        self.generate(num_servers, num_switches, num_ports)

    def generate(self, num_servers, num_switches, num_ports):

        # TODO: code for generating the jellyfish topology
        # each switch i has k_i ports, where r_i is the connection to other switches and remaining k_i - r_i ports are used to connect to servers
        # degree is number of ports per switch
        degree = num_switches * num_ports // num_servers
        max_server_per_switch = num_ports - degree
        server_per_switch = - (num_servers // - num_switches)

        edge_cnt = 0

        for switch_id in range(num_switches):
            switch_node = Node(switch_id, "switch")

            if(edge_cnt < num_servers):
                for server_id in range(server_per_switch):
                    if(edge_cnt + server_id == num_servers):
                        break
                    server_node = Node(server_id + edge_cnt, "server")
                    switch_node.add_edge(server_node)
                    self.servers.append(server_node)

            self.switches.append(switch_node)
            edge_cnt += server_per_switch

        # # Initialize servers and top of the rack switches
        # while(edge_cnt < num_servers):
        # 	num_connections = random.randrange(1, max_server_per_switch)

        # 	switch_node = Node(switch_id, "tor_switch")

        # 	edg = 0
        # 	if(num_connections + edge_cnt >= num_servers):
        # 		edg = num_connections + edge_cnt - num_servers

        # 	for tmp in range(num_connections - edg):
        # 		server_node = Node(tmp + edge_cnt, "server")
        # 		switch_node.add_edge(server_node)
        # 		self.servers.append(server_node)
        # 		if(num_connections + tmp == num_servers):
        # 			break

        # 	self.switches.append(switch_node)

        # 	switch_id += 1
        # 	edge_cnt += num_connections

        # # Add remaining switches without any connections to the dict
        # while(switch_id < num_switches):
        # 	self.switches.append(Node(switch_id, "net_switch"))
        # 	switch_id += 1

        switches_to_connect = len(self.switches)

        # Add free switches randomly to the network
        while(switches_to_connect > 2):
            while(True):
                node1 = random.choice(self.switches)
                node2 = random.choice(self.switches)
                if((len(node1.edges) >= num_ports) or (len(node2.edges) >= num_ports) or node1.is_neighbor(node2)):
                    continue
                else:
                    break

            node1.add_edge(node2)
            if(len(node1.edges) >= num_ports):
                switches_to_connect -= 1
            if(len(node2.edges) >= num_ports):
                switches_to_connect -= 1

        remaining_switches = []
        for switch_id in range(num_switches):
            switch = self.switches[switch_id]
            if(len(switch.edges) < num_ports):
                remaining_switches.append(switch)

        # Connect last free switches to network until only one switch with one open port remains
        while(len(remaining_switches) > 1):

            tmp = random.choice(self.switches)
            sw_to_connect = remaining_switches[0]

            if(tmp == sw_to_connect):
                continue

            edg_to_remove = random.choice(tmp.edges)
            sw_to_connect.add_edge(edg_to_remove.lnode)
            sw_to_connect.add_edge(edg_to_remove.rnode)

            tmp.remove_edge(edg_to_remove)

            if(len(sw_to_connect.edges) >= num_ports):
                remaining_switches.remove(sw_to_connect)

        # self.plot()

    def plot(self):
        nodes = []
        edgs = []
        for i in range(10):
            nd = random.choice(self.switches)
            if(nd not in nodes):
                nodes.append(nd)

        for i in range(len(nodes)):
            nd = nodes[i]
            for j in range(len(nd.edges)):
                tpl = (nd.edges[j].lnode.id, nd.edges[j].rnode.id)
                edgs.append(tpl)

        JF_GRAPH = nx.Graph(edgs)
        pos = nx.spring_layout(JF_GRAPH)
        nx.draw(JF_GRAPH, pos, with_labels=True)
        plt.savefig("jf.png")


class Fattree:

    def __init__(self, num_ports):
        self.servers = []
        self.switches = []
        self.generate(num_ports)

    def generate(self, num_ports):
        # TODO: code for generating the fat-tree topology
        PLOT_NODE_SPACING = 2

        num_pods = num_ports
        num_hosts = (num_pods ** 3) / 4
        num_switches_per_layer = num_pods // 2
        num_core_switches = (num_pods // 2) ** 2

        plot_nodes = {}
        plot_core_sw_x = (num_pods * num_switches_per_layer * PLOT_NODE_SPACING -
                          PLOT_NODE_SPACING * 1.5) / 2 - (num_core_switches / 2 - 1)
        plot_agg_sw_x_offset = 0

        # core_switches = {}

        pod_switches = {}
        # fig, ax = plt.subplots(figsize=(num_pods * num_switches_per_layer * PLOT_NODE_SPACING * 2, num_core_switches))
        # ax.set_xlim(-1, num_pods * num_switches_per_layer * PLOT_NODE_SPACING)
        # ax.set_ylim(0, 4)
        # plt.axis('off')

        for sw_core_row_idx in range(0, num_core_switches // num_switches_per_layer):
            core_switches = []
            for sw_core_column_idx in range(0, num_switches_per_layer):
                core_sw_id = "10.{}.{}.{}".format(
                    num_pods, sw_core_row_idx + 1, sw_core_column_idx + 1)
                core_sw = Node(core_sw_id, "core-sw")
                core_switches.append(core_sw)
                self.switches.append(core_sw)
                # plot_core_sw = ax.annotate(core_sw_id, xy=(plot_core_sw_x, 4), xycoords="data",
                # 					va="center", ha="center",
                # 					bbox=dict(boxstyle="round", fc="w"))
                # plot_nodes[core_sw_id] = plot_core_sw
                # plot_core_sw_x += 1

            for pod_idx in range(0, num_pods):
                pod_switches.setdefault(pod_idx, {"agg-sw": [], "edge-sw": []})

                agg_sw_id = "10.{}.{}.{}".format(
                    pod_idx, num_switches_per_layer + sw_core_row_idx, 1)
                agg_sw = Node(agg_sw_id, "agg-sw")

                # plot_agg_sw_base_x = pod_idx * num_switches_per_layer * PLOT_NODE_SPACING

                # plot_agg_sw = ax.annotate(agg_sw_id, xy=(plot_agg_sw_base_x + plot_agg_sw_x_offset, 3), xycoords="data",
                # 				va="center", ha="center",
                # 				bbox=dict(boxstyle="round", fc="w"))
                # plot_nodes[agg_sw_id] = plot_agg_sw

                for core_sw in core_switches:
                    core_sw.add_edge(agg_sw)
                    # con = ConnectionPatch(plot_nodes[core_sw.id].xy, plot_agg_sw.xy, "data", "data", arrowstyle="->", shrinkA=2, shrinkB=2,
                    # 		mutation_scale=1, fc="w")
                    # ax.add_artist(con)

                pod_switches[pod_idx]["agg-sw"].append(agg_sw)
                self.switches.append(agg_sw)

                edge_sw_id = "10.{}.{}.{}".format(pod_idx, sw_core_row_idx, 1)
                edge_sw = Node(edge_sw_id, "edge-sw")

                # plot_edge_sw = ax.annotate(edge_sw_id, xy=(plot_agg_sw_base_x + plot_agg_sw_x_offset, 2), xycoords="data",
                # 		va="center", ha="center",
                # 		bbox=dict(boxstyle="round", fc="w"))
                # plot_nodes[edge_sw_id] = plot_edge_sw

                # agg_sw.add_edge(edge_sw)
                # edge_sw.add_edge(agg_sw)
                pod_switches[pod_idx]["edge-sw"].append(edge_sw)
                self.switches.append(edge_sw)

                for srv_idx in range(1, (num_pods // 2) + 1):
                    srv_id = "10.{}.{}.{}".format(
                        pod_idx, sw_core_row_idx, srv_idx + 1)
                    srv = Node(srv_id, "server")
                    # plot_srv = ax.annotate(srv_id, xy=(-1.5 + srv_idx + plot_agg_sw_base_x + plot_agg_sw_x_offset, 1), xycoords="data",
                    # 		va="center", ha="center",
                    # 		bbox=dict(boxstyle="round", fc="w"))
                    # con = ConnectionPatch(plot_nodes[edge_sw_id].xy, plot_srv.xy, "data", "data", arrowstyle="->", shrinkA=2, shrinkB=2,
                    # 		mutation_scale=1, fc="w")
                    # ax.add_artist(con)
                    self.servers.append(srv)
                    edge_sw.add_edge(srv)

            plot_agg_sw_x_offset += PLOT_NODE_SPACING

        for pod in pod_switches:
            for agg_sw in pod_switches[pod]["agg-sw"]:
                for edge_sw in pod_switches[pod]["edge-sw"]:
                    agg_sw.add_edge(edge_sw)
                    # con = ConnectionPatch(plot_nodes[agg_sw.id].xy, plot_nodes[edge_sw.id].xy, "data", "data", arrowstyle="->", shrinkA=2, shrinkB=2,
                    # 		mutation_scale=1, fc="w")
                    # ax.add_artist(con)

        # fig.savefig("test.png")


def dijkstra(start_node, switches, servers):
    # unvisited = {**{node: {"weight": None, "path": []} for node in switches}, **{node: {"weight": None, "path": []} for node in servers}}
    # unvisited = {node: {"weight": None, "path": []} for node in switches}
    # unvisited[start_node] = {"weight": 0, "path": []}
    unvisited = {node: None for node in switches}
    # unvisited = {**{node: None for node in switches}, **{node: None for node in servers}}
    unvisited[start_node] = 0
    path = {}
    visited = []

    unique = count()
    pqueue = queue.PriorityQueue()
    pqueue.put((0, next(unique), start_node))

    while not pqueue.empty():
        current_node = pqueue.get()[2]
        visited.append(current_node)

        # check switches to find next possible hop which was not yet visited
        for neighbor in (switches):
            if current_node.is_neighbor(neighbor) and neighbor not in visited:
                # check if
                # old_weight = unvisited[neighbor]["weight"]
                old_weight = unvisited[neighbor]

                # use hard-coded 1 because we have no weighted links
                # new_weight = unvisited[current_node]["weight"] + 1
                new_weight = unvisited[current_node] + 1

                if old_weight is None or new_weight < old_weight:
                    pqueue.put((new_weight, next(unique), neighbor))
                    # unvisited[neighbor]["weight"] = new_weight
                    unvisited[neighbor] = new_weight
                    # unvisited[neighbor]["path"] = unvisited[current_node]["path"].copy()
                    # unvisited[neighbor]["path"].append(current_node)
                    # save path for reconstruction
                    path[neighbor] = current_node
        pqueue.task_done()
    return unvisited, path
