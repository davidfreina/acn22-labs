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

import queue
import random
import sys
from itertools import count
from operator import itemgetter
from time import time
from math import log
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

# Class for an edge in the graph


class Edge:
    def __init__(self):
        self.lnode = None
        self.rnode = None

    def __repr__(self):
        return "Edge({}, {})".format(self.lnode.id, self.rnode.id)
        # return "\n\t\tEdge({}, {})".format(self.lnode.id, self.rnode.id)

    def __str__(self):
        return "Edge({}, {})".format(self.lnode.id, self.rnode.id)
        # return "\n\t\tEdge({}, {})".format(self.lnode.id, self.rnode.id)

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
        return "Node({}, {})".format(self.id, self.type, self.edges)
        # return "\tNode({}, {}, {})\n".format(self.id, self.type, self.edges)

    def __str__(self):
        return "Node({})".format(self.id, self.type, self.edges)
        # return "\tNode({}, {}, {})\n".format(self.id, self.type, self.edges)

    def __eq__(self, other):
        return (self.id == other.id and self.type == other.type)

    def __hash__(self):
        return hash(str(self.id) + str(self.type))

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

    def get_edge(self, node):
        for edge in self.edges:
            if node in (edge.lnode, edge.rnode):
                return edge
        return None

    # Returns bcube for server and level for switch
    def get_bcube(self):
        if self.type == "server":
            return int(self.id.split('.')[0])
        return int(self.id.split('.')[0])


class Jellyfish:

    def __init__(self, num_servers, num_switches, num_ports, plot=False):
        self.servers = []
        self.switches = []
        self.generate(num_servers, num_switches, num_ports, plot)

    def generate(self, num_servers, num_switches, num_ports, plot):

        # TODO: code for generating the jellyfish topology
        server_per_switch = - (num_servers // - num_switches)
        edge_cnt = 0

        # Distributing the servers over the switches
        for switch_id in range(num_switches):
            switch_node = Node(switch_id, "switch")

            if edge_cnt < num_servers:
                for server_id in range(server_per_switch):
                    if edge_cnt + server_id == num_servers:
                        break
                    server_node = Node(server_id + edge_cnt, "server")
                    switch_node.add_edge(server_node)
                    self.servers.append(server_node)

            self.switches.append(switch_node)
            edge_cnt += server_per_switch

        switches_to_connect = self.switches.copy()
        switches_connected = []
        retries = 0

        # Add free switches randomly to the network (retry necessary to prevent looping forever)
        while len(switches_to_connect) > 2 and retries < num_switches * 4:
            node1 = random.choice(switches_to_connect)
            node2 = random.choice(switches_to_connect)

            if node1 == node2:
                retries += 1
                continue

            if len(node1.edges) < num_ports and len(node2.edges) < num_ports and not node1.is_neighbor(node2):

                node1.add_edge(node2)

                if(len(node1.edges) == num_ports):
                    switches_to_connect.remove(node1)
                    switches_connected.append(node1)
                if(len(node2.edges) == num_ports):
                    switches_to_connect.remove(node2)
                    switches_connected.append(node2)

        if not switches_to_connect:
            if plot:
                self.plot(num_ports)
            return

        while len(switches_to_connect) > 1:
            avail_sw = switches_to_connect[0]

            if len(avail_sw.edges) is num_ports - 1:
                switches_to_connect.remove(avail_sw)
                switches_connected.append(avail_sw)
                continue

            tmp = random.choice(switches_connected)
            rm_edg = random.choice(tmp.edges)
            lnod = rm_edg.lnode
            rnod = rm_edg.rnode

            if not (avail_sw.is_neighbor(lnod) or avail_sw.is_neighbor(rnod)):
                rm_edg.remove()
                avail_sw.add_edge(lnod)
                avail_sw.add_edge(rnod)

                if len(avail_sw.edges) == num_ports:
                    switches_to_connect.remove(avail_sw)
                    switches_connected.append(avail_sw)

        if len(switches_to_connect) == 1 or len(switches_to_connect) == 0:
            switches_connected.append(switches_to_connect.pop())
        else:
            print("error")
            sys.exit(1)

        switches_connected.sort(key=lambda switch: switch.id)
        self.switches = switches_connected

        # for i in self.switches:
        #     if len(i.edges) != num_ports and len(i.edges) != num_ports - 1:
        #         print(i)
        #         print(len(i.edges))

        if plot:
            self.plot(num_ports)

    def plot(self, num_ports):
        fig, ax = plt.subplots(
            subplot_kw={'projection': 'polar'}, figsize=(20, 20))
        plt.axis('off')
        ax.set_rmax(5)
        ax.set_rticks([])

        switches = self.switches.copy()
        servers = self.servers.copy()

        tor_switches = []
        for server in servers:
            if len(server.edges) != 1:
                print('too many switches')
                sys.exit()

            edge = server.edges[0]

            if server == edge.rnode:
                if edge.lnode not in tor_switches:
                    tor_switches.append(edge.lnode)
            else:
                if edge.rnode not in tor_switches:
                    tor_switches.append(edge.rnode)

        tor_switch_offset = np.pi * 2 / len(tor_switches)
        full_circle = np.pi * 2

        server_offset = tor_switch_offset / len(tor_switches[0].edges)

        switch_plots = {}
        done_edges = []

        while full_circle > 0 and tor_switches:
            full_circle -= tor_switch_offset
            current_switch = tor_switches.pop(0)
            switches.remove(current_switch)
            current_switch_plot = ax.annotate(current_switch.id, xy=(full_circle, 4),
                                              va="center", ha="center",
                                              bbox=dict(boxstyle="round", fc="w"))
            switch_plots[current_switch.id] = current_switch_plot

            for idx, edge in enumerate(current_switch.edges):
                if edge.rnode == current_switch and edge.lnode.type == "server":
                    done_edges.append(
                        tuple((edge.lnode.id, current_switch.id)))
                    current_server_plot = ax.annotate(edge.lnode.id, xy=(full_circle - server_offset * idx, 5),
                                                      va="center", ha="center",
                                                      bbox=dict(boxstyle="round", fc="w"))
                elif edge.rnode.type == "server":
                    done_edges.append(
                        tuple((edge.rnode.id, current_switch.id)))
                    current_server_plot = ax.annotate(edge.rnode.id, xy=(full_circle - server_offset * idx, 5),
                                                      va="center", ha="center",
                                                      bbox=dict(boxstyle="round", fc="w"))

                con = ConnectionPatch(current_switch_plot.xy, current_server_plot.xy, "data", "data", arrowstyle="-", shrinkA=2, shrinkB=2,
                                      mutation_scale=1, fc="w")
                ax.add_artist(con)

        switch_offset = np.pi * 2 / len(switches)
        full_circle = np.pi * 2

        while switches:
            full_circle -= switch_offset
            current_switch = switches.pop(0)
            current_switch_plot = ax.annotate(current_switch.id, xy=(full_circle, 3),
                                              va="center", ha="center",
                                              bbox=dict(boxstyle="round", fc="w"))
            switch_plots[current_switch.id] = current_switch_plot

        for idx, switch_plot in enumerate(switch_plots):
            switch = self.switches[switch_plot]
            for edge in switch.edges:
                if edge.rnode.type != "server" and edge.lnode.type != "server" and tuple((edge.rnode.id, edge.lnode.id)) not in done_edges and tuple((edge.lnode.id, edge.rnode.id)) not in done_edges:
                    if edge.rnode.id == switch_plot:
                        switch2_plot = switch_plots[edge.lnode.id]
                    else:
                        switch2_plot = switch_plots[edge.rnode.id]

                    con = ConnectionPatch(switch_plots[switch_plot].xy, switch2_plot.xy, "data", "data", arrowstyle="-", shrinkA=2, shrinkB=2,
                                          mutation_scale=1, fc="w")
                    ax.add_artist(con)

        plt.savefig("plot_jellyfish_{}.png".format(num_ports))
        plt.clf()


class Fattree:

    def __init__(self, num_ports, plot=False):
        self.servers = []
        self.switches = []
        self.generate(num_ports, plot)

    def generate(self, num_ports, plot):
        # TODO: code for generating the fat-tree topology
        PLOT_NODE_SPACING = 2

        num_pods = num_ports
        num_switches_per_layer = num_pods // 2
        num_core_switches = (num_pods // 2) ** 2

        plot_nodes = {}
        plot_core_sw_x = (num_pods * num_switches_per_layer * PLOT_NODE_SPACING -
                          PLOT_NODE_SPACING * 1.5) / 2 - (num_core_switches / 2 - 1)
        plot_agg_sw_x_offset = 0

        pod_switches = {}

        if plot:
            fig, ax = plt.subplots(figsize=(
                num_pods * num_switches_per_layer * PLOT_NODE_SPACING * 2, num_core_switches))
            ax.set_xlim(-1, num_pods * num_switches_per_layer *
                        PLOT_NODE_SPACING)
            ax.set_ylim(0, 4)
            plt.axis('off')

        for sw_core_row_idx in range(0, num_core_switches // num_switches_per_layer):
            core_switches = []
            for sw_core_column_idx in range(0, num_switches_per_layer):
                core_sw_id = "10.{}.{}.{}".format(
                    num_pods, sw_core_row_idx + 1, sw_core_column_idx + 1)
                core_sw = Node(core_sw_id, "core-sw")
                core_switches.append(core_sw)
                self.switches.append(core_sw)
                if plot:
                    plot_core_sw = ax.annotate(core_sw_id, xy=(plot_core_sw_x, 4), xycoords="data",
                                               va="center", ha="center",
                                               bbox=dict(boxstyle="round", fc="w"))
                    plot_nodes[core_sw_id] = plot_core_sw
                    plot_core_sw_x += 1

            for pod_idx in range(0, num_pods):
                pod_switches.setdefault(pod_idx, {"agg-sw": [], "edge-sw": []})

                agg_sw_id = "10.{}.{}.{}".format(
                    pod_idx, num_switches_per_layer + sw_core_row_idx, 1)
                agg_sw = Node(agg_sw_id, "agg-sw")

                if plot:
                    plot_agg_sw_base_x = pod_idx * num_switches_per_layer * PLOT_NODE_SPACING

                    plot_agg_sw = ax.annotate(agg_sw_id, xy=(plot_agg_sw_base_x + plot_agg_sw_x_offset, 3), xycoords="data",
                                              va="center", ha="center",
                                              bbox=dict(boxstyle="round", fc="w"))
                    plot_nodes[agg_sw_id] = plot_agg_sw

                for core_sw in core_switches:
                    core_sw.add_edge(agg_sw)
                    if plot:
                        con = ConnectionPatch(plot_nodes[core_sw.id].xy, plot_agg_sw.xy, "data", "data", arrowstyle="->", shrinkA=2, shrinkB=2,
                                              mutation_scale=1, fc="w")
                        ax.add_artist(con)

                pod_switches[pod_idx]["agg-sw"].append(agg_sw)
                self.switches.append(agg_sw)

                edge_sw_id = "10.{}.{}.{}".format(pod_idx, sw_core_row_idx, 1)
                edge_sw = Node(edge_sw_id, "edge-sw")

                if plot:
                    plot_edge_sw = ax.annotate(edge_sw_id, xy=(plot_agg_sw_base_x + plot_agg_sw_x_offset, 2), xycoords="data",
                                               va="center", ha="center",
                                               bbox=dict(boxstyle="round", fc="w"))
                    plot_nodes[edge_sw_id] = plot_edge_sw

                pod_switches[pod_idx]["edge-sw"].append(edge_sw)
                self.switches.append(edge_sw)

                for srv_idx in range(1, (num_pods // 2) + 1):
                    srv_id = "10.{}.{}.{}".format(
                        pod_idx, sw_core_row_idx, srv_idx + 1)
                    srv = Node(srv_id, "server")

                    if plot:
                        plot_srv = ax.annotate(srv_id, xy=(-1.5 + srv_idx + plot_agg_sw_base_x + plot_agg_sw_x_offset, 1), xycoords="data",
                                               va="center", ha="center",
                                               bbox=dict(boxstyle="round", fc="w"))
                        con = ConnectionPatch(plot_nodes[edge_sw_id].xy, plot_srv.xy, "data", "data", arrowstyle="->", shrinkA=2, shrinkB=2,
                                              mutation_scale=1, fc="w")
                        ax.add_artist(con)

                    self.servers.append(srv)
                    edge_sw.add_edge(srv)

            plot_agg_sw_x_offset += PLOT_NODE_SPACING

        for pod in pod_switches:
            for agg_sw in pod_switches[pod]["agg-sw"]:
                for edge_sw in pod_switches[pod]["edge-sw"]:
                    agg_sw.add_edge(edge_sw)
                    if plot:
                        con = ConnectionPatch(plot_nodes[agg_sw.id].xy, plot_nodes[edge_sw.id].xy, "data", "data", arrowstyle="->", shrinkA=2, shrinkB=2,
                                              mutation_scale=1, fc="w")
                        ax.add_artist(con)

        if plot:
            fig.savefig("plot_fattree_{}.png".format(num_ports))
            fig.clf()
            plt.clf()


class Bcube:
    def __init__(self, num_servers, num_ports, plot=False):
        self.servers = []
        self.switches = []

        k = int((log(num_servers) - log(num_ports)) / log(num_ports))

        if num_ports * num_ports ** k < num_servers:
            print('invalid input detected')
            print(
                'with {} ports per bcube_0 switch only {} total servers will be available'.format(num_ports, num_ports * num_ports ** k))

        self.generate(num_ports, k, plot)

    def generate(self, num_server_bcube, num_bcube, plot):

        num_servers = num_server_bcube ** (num_bcube + 1)
        for i in range(num_servers):
            server = Node('{}.{}'.format(i // num_server_bcube, i %
                          num_server_bcube), "server")
            self.servers.append(server)

        for level in range(num_bcube + 1):
            lower = num_server_bcube ** level
            upper = num_server_bcube ** (level + 1)
            for i in range(num_server_bcube ** num_bcube):
                switch = Node('{}.{}'.format(level, i),
                              "switch-{}".format(level))
                self.switches.append(switch)
                start = i % lower + i // lower * upper
                hosts = range(start, start + upper, lower)
                for v in hosts:
                    switch.add_edge(self.servers[v])
        if plot:
            self.plot(num_bcube, num_server_bcube,
                      (num_server_bcube ** num_bcube), num_servers)
        return

    def plot(self, k, num_servers_per_bcube, switches_per_level, num_servers):
        fig, ax = plt.subplots(figsize=(num_servers, 10))
        ax.set_xlim(0, len(self.servers))
        ax.set_ylim(0, k+1)
        plt.axis('off')

        switch_start = num_servers / 2 - switches_per_level + 1

        plotted = []
        plotted_nodes = {}
        for switch in self.switches:
            switch_level = int(switch.type.split("-")[1])
            position = int(switch.id.split(".")[1])
            plotted_nodes[switch] = ax.annotate(switch.id, xy=(switch_start + position, switch_level + 1), xycoords="data",
                                                va="center", ha="center",
                                                bbox=dict(boxstyle="round", fc="w"))

        for server in self.servers:
            split = server.id.split(".")
            plotted_server = ax.annotate(server.id, xy=(int(split[0]) * num_servers_per_bcube + int(split[1]), 0), xycoords="data",
                                         va="center", ha="center",
                                         bbox=dict(boxstyle="round", fc="w"))

            for edge in server.edges:
                for node in [edge.rnode, edge.lnode]:
                    if node == server:
                        continue
                    con = ConnectionPatch(plotted_server.xy, plotted_nodes[node].xy, "data", "data", arrowstyle="-", shrinkA=2, shrinkB=2,
                                          mutation_scale=1, fc="w")
                    ax.add_artist(con)

        fig.savefig("plot_bcube_{}.png".format(num_servers))
        fig.clf()
        plt.clf()


class Dcell:
    # num_ports = n, level = n + 1 (with n > 0)
    def __init__(self, num_ports, plot=False):
        self.servers = []
        self.switches = []
        self.dcells = []
        self.plot_edges = []
        self.edge_cnt = 0
        self.level = 1
        self.rec_cnt = 0
        # offset[0] = (numbers of cells on level = index, numbers of servers on level = index)
        self.offsets = []
        self.g_l = 1
        self.num_ports = num_ports
        self.t_previous = num_ports

        lv = 1
        while lv <= self.level:
            self.g_l = self.g_l * (num_ports + 1)
            self.t_previous = self.g_l * num_ports
            lv += 1

        self.generate(num_ports, self.level)

        for i in range(len(self.servers)):
            for j in range(len(self.servers[i].edges)):
                edge_tuple = (
                    self.servers[i].edges[j].lnode.id, self.servers[i].edges[j].rnode.id)
                self.plot_edges.append(edge_tuple)

        print(self.plot_edges)
        if plot:
            self.plot()

    def generate(self, num_ports, level, prefix=None):

        if prefix is None:
            prefix = (0, num_ports)

        # Building Dcell0
        if level == 0:
            miniswitch = Node(self.rec_cnt, "switch")
            for i in range(num_ports):
                server_id = "{}.{}".format(self.rec_cnt, i)
                server = Node(server_id, "server")
                miniswitch.add_edge(server)
                self.servers.append(server)
                self.edge_cnt += 1
            self.switches.append(miniswitch)
            self.rec_cnt += 1
            return

        for j in range(self.g_l):
            self.generate(num_ports, level - 1, prefix=(j, num_ports))

        for dcell in range(self.t_previous):
            for server in range(dcell + 1, self.g_l):
                index1 = "{}.{}".format(dcell, server - 1)
                index2 = "{}.{}".format(server, dcell)
                node1 = self.find_node(index1)
                node2 = self.find_node(index2)
                if node1 is None or node2 is None:
                    print("Node could not be found.")
                    sys.exit()
                node1.add_edge(node2)
        return

    def find_node(self, index):
        # print(index)
        for node in self.servers:
            if node.id == index:
                return node
        # print(self.servers)

        return None

    def generate_seq(self, num_ports, level, plot=False):

        if self.level == 0:
            miniswitch = Node(0, "switch")
            for c0 in range(num_ports):
                server_id = "{}.{}".format(self.rec_cnt, c0)
                server = Node(server_id, "server")
                miniswitch.add_edge(server)
                self.servers.append(server)
                self.edge_cnt += 1
            self.switches.append(miniswitch)
            return

        lv = 1
        self.offsets.append(tuple((0, num_ports)))
        while lv <= level:
            self.g_l = self.g_l * (num_ports + 1)
            self.t_previous = self.g_l * num_ports
            cell_offset = self.g_l
            server_offset = self.t_previous
            offset = (cell_offset, server_offset)
            self.offsets.append(offset)
            lv += 1

        for dcell in range(self.g_l):
            miniswitch = Node(dcell, "switch")
            for serv in range(num_ports):
                server_id = "{}.{}".format(dcell, serv)
                server = Node(server_id, "server")
                miniswitch.add_edge(server)
                self.servers.append(server)

        for lev in range(level + 1):
            dc = Node(lev, "dcell")
            offs = 1
            for ser in range(self.offsets[lev][offs]):
                dc.edges.append(ser)
            self.dcells.append(dc)

        for i in range(len(self.dcells)):
            print(self.dcells[i].edges)

        test = 0
        while test < 20:
            self.connect_nodes(test, test + 4, test + 5)
            test += 5

    def connect_nodes(self, tl_start, tl_end, gl_end):
        tl_start = self.offsets[0][0]
        tl_end = self.offsets[1][1]
        gl_end = self.offsets[1][0]

        for i in range(tl_start, tl_end):
            for j in range(0, gl_end):

                uid_1, uid_2 = j, i
                index1 = "{}.{}".format(i, uid_1)
                index2 = "{}.{}".format(j, uid_2)
                node1 = self.find_node(index1)
                node2 = self.find_node(index2)
                if node1 is None or node2 is None:
                    print("Node could not be found.")
                    sys.exit()
                node1.add_edge(node2)

            print("Cells: {}".format(i))
            print("Server: {}".format(j))
            print("---------------")

        # def connect_dcells(self, num_ports):
        #     print(self.servers)
        #     for dcell in range(self.t_previous):
        #         server = dcell + 1
        #         while server < num_ports:

        #             index1 = "{}.{}".format(dcell, server - 1)
        #             index2 = "{}.{}".format(server, dcell)
        #             node1 = self.find_node(index1)
        #             node2 = self.find_node(index2)
        #             if node1 is None or node2 is None:
        #                 print("Node could not be found.")
        #                 sys.exit()
        #             node1.add_edge(node2)
        #             server += 1

    def plot(self):
        G = nx.Graph()
        G.add_edges_from(self.plot_edges)
        nx.draw(G, with_labels=True, pos=nx.spring_layout(G), font_weight='bold')
        plt.savefig("plot_dcell_{}.png".format(self.num_ports))
        plt.clf()


def dijkstra(start_node, switches, end_node=None):
    # unvisited = {**{node: {"weight": None, "path": []} for node in switches}, **{node: {"weight": None, "path": []} for node in servers}}
    unvisited = {node: {"distance": None, "path": []} for node in switches}
    unvisited[start_node] = {"distance": 0, "path": []}
    # unvisited = {node: None for node in switches}
    # unvisited[start_node] = 0
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
                old_weight = unvisited[neighbor]["distance"]
                # old_weight = unvisited[neighbor]

                # use hard-coded 1 because we have no weighted links
                new_weight = unvisited[current_node]["distance"] + 1
                # new_weight = unvisited[current_node] + 1

                if old_weight is None or new_weight < old_weight:
                    pqueue.put((new_weight, next(unique), neighbor))
                    unvisited[neighbor]["distance"] = new_weight
                    # unvisited[neighbor] = new_weight
                    unvisited[neighbor]["path"] = unvisited[current_node]["path"].copy()
                    unvisited[neighbor]["path"].append(current_node)

                    if neighbor is end_node:
                        pqueue.task_done()
                        for node in unvisited:
                            unvisited[node]["path"].append(node)
                        return unvisited

                    # save path for reconstruction
                    # path[neighbor] = current_node

        pqueue.task_done()

    for node in unvisited:
        unvisited[node]["path"].append(node)

    return unvisited


def find_path(start_node, end_node, switches):
    distances = dijkstra(start_node, switches, end_node)

    if distances[end_node] is None:
        return {"distance": None, "path": []}

    return distances[end_node]


def ksp_yen(start_node, end_node, switches, max_k=2):
    print("start yen for start: {}, end: {}".format(start_node, end_node))
    start = time()
    distances = dijkstra(start_node, switches)

    A = [distances[end_node]]
    B = []

    if not A[0]["path"]:
        return A

    for k in range(1, max_k):
        k_shortest_path = A[-1]
        for i in range(0, k_shortest_path["distance"]):
            spur_node = k_shortest_path["path"][i]
            root_path = k_shortest_path["path"][:i + 1]

            # print(spur_node)
            # print(root_path)

            edges_removed = []
            for path in A:
                curr_path = path["path"]
                # if len(curr_path) > i and root_path == curr_path[:i + 1]:

                # print(curr_path)
                # print(root_path)
                # print(root_path == curr_path[:i + 1])

                if root_path == curr_path[:i + 1]:
                    edge = curr_path[i].get_edge(curr_path[i + 1])
                    # print(edge)

                    if edge:
                        edge.remove()
                        edges_removed.append([curr_path[i], curr_path[i + 1]])
                    # curr_path[i].remove_edge(edge)
                    # curr_path[i + 1].remove_edge(edge)

            for node in root_path:
                if node is not spur_node:
                    # print(node)
                    for edge in node.edges:
                        edges_removed.append([edge.rnode, edge.lnode])
                        edge.remove()

            path_spur = find_path(spur_node, end_node, switches)
            # print("path_spur: {}".format(path_spur))

            if path_spur["distance"]:
                path_total = root_path[:-1] + path_spur["path"]
                distance_total = distances[spur_node]["distance"] + \
                    path_spur["distance"]
                potential_path = {
                    "distance": distance_total, "path": path_total}
                # print(potential_path)

                if potential_path not in B:
                    B.append(potential_path)
                    # print("add to B")

            for edge in edges_removed:
                edge[0].add_edge(edge[1])

        B = sorted(B, key=itemgetter("distance"))
        while len(B) != 0:
            next_path = B.pop(0)
            if next_path not in A:
                # print("add to A")
                # Adding a shortcut if more than 8 paths have been found
                # and the next path is longer than the last one we can
                # simply stop the execution because ECMP 8 and 64 are done
                A.append(next_path)
                if k > 6:
                    if next_path["distance"] > A[0]["distance"]:
                        print("Stopped Yen after k: {}, time: {}s, paths: {}".format(
                            k, time() - start, len(A)))
                        return A
                break

    print("Stopped Yen after k: {}, time: {}s".format(k, time() - start))
    return A
