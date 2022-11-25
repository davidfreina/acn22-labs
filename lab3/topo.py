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

# import matplotlib.pyplot as plt
# from matplotlib.patches import ConnectionPatch
# import numpy as np

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


class Fattree:

    def __init__(self, num_ports, plot=False):
        self.servers = []
        self.switches = []
        self.generate(num_ports, plot)

    def generate(self, num_ports, plot):
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


# if __name__ == "__main__":
#     fatty = Fattree(4)
#     net = Mininet(topo=fatty)
#     net.start()
#     net.waitConnected()
#     net.pingAll()
#     print("finished pinging")
