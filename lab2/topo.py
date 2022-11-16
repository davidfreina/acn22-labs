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

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import ConnectionPatch

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


class Jellyfish:

    def __init__(self, num_servers, num_switches, num_ports, plot = False):
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



        # cnt = 0
        # for i in self.switches:
        #     if len(i.edges) != num_ports:
        #         cnt += 1
        # print(cnt)
        # print("----------------")
        # print(len(switches_to_connect))
        # Connect last free switches to network until only one switch with one open port remains


        # Adding switches with >= 2 free ports
        # for avail in self.switches:
        #     if num_ports - len(avail.edges) >= 2 and avail not in switches_to_connect:
        #         switches_to_connect.add(avail)
        # print(switches_to_connect)
        # for i in self.switches:
        #     if len(i.edges) != num_ports:
        #         print(i)
        #         print(len(i.edges))
        # print("-----------------")


        # if (len(switches_to_connect) == 1) and (len(switches_to_connect[0].edges) == num_ports - 1):
        #     return

        to_remove = []
        for switch in switches_to_connect:
            if len(switch.edges) == num_ports - 1:
                to_remove.append(switch)
                switches_connected.append(switch)
        for switch in to_remove:
            switches_to_connect.remove(switch)


        if not switches_to_connect:
            return

        while switches_to_connect:
            avail_sw = switches_to_connect[0]
            # if num_ports - len(avail_sw.edges) < 2:
            #     print("Removing: {}".format(avail_sw))
            #     switches_to_connect.remove(avail_sw)
            #     continue

            while True:
                tmp = random.choice(switches_connected)
                if tmp != avail_sw and not tmp.is_neighbor(avail_sw) and tmp not in switches_to_connect:
                    break

            # print("avail_sw: {}".format(avail_sw))
            # print("len(avail_sw.edges): {}".format(len(avail_sw.edges)))
            # print(avail_sw.edges)
            # print(len(tmp.edges))

            rm_edg = random.choice(tmp.edges)
            lnod = rm_edg.lnode
            rnod = rm_edg.rnode

            # print(rm_edg)
            # print(lnod)
            # print(rnod)


            if not (avail_sw.is_neighbor(lnod) or avail_sw.is_neighbor(rnod)):
                #print("Tmp: {}".format(tmp))
                #print("RM Edg: {}".format(rm_edg))

                avail_sw.add_edge(lnod)
                avail_sw.add_edge(rnod)
                # self.switches[avail_sw.id].add_edge(lnod)
                # self.switches[avail_sw.id].add_edge(rnod)

                # print(avail_sw.edges)

                # if rm_edg in avail_sw.edges:
                #     avail_sw.remove_edge(rm_edg)
                # if rm_edg in self.switches[avail_sw.id].edges:
                #     self.switches[avail_sw.id].remove_edge(rm_edg)

                # Does not need to be removed in switches_to_connect
                # because tmp is not in switches_to_connect
                # therefore the link can also not exist there
                rm_edg.remove()
                # print(len(tmp.edges))


                # print(len(avail_sw.edges))
                if len(avail_sw.edges) == num_ports or len(avail_sw.edges) == num_ports - 1:
                    switches_to_connect.remove(avail_sw)
                    switches_connected.append(avail_sw)

                # for i in self.switches:
                #     if num_ports - len(i.edges) >= 2 and i not in switches_to_connect:
                #         switches_to_connect.add(i)






        # # If still switch(es) with >= 2 free ports are available
        # if switches_to_connect:

        #     for i in switches_to_connect:
        #         avail_sw = switches_to_connect[i]
        #         print(len(avail_sw.edges))
        #         while True:
        #             tmp = random.choice(self.switches)
        #             if tmp != avail_sw: break
        #         print("-----------------")
        #         rm_edg = random.choice(tmp.edges)
        #         tmp.remove_edge(rm_edg)

        #         print(rm_edg)
        #         print(rm_edg.lnode)
        #         print(rm_edg.rnode)

        #         print(len(tmp.edges))


        if switches_to_connect:
            print("not fully connected")

        self.switches = switches_connected




        print("-----------------")
        # while len(switches_to_connect) > 3:

        #     tmp = random.choice(self.switches)
        #     sw_to_connect = switches_to_connect[0]
        #     print(len(sw_to_connect.edges))

        #     if tmp == sw_to_connect:
        #         continue

        #     edg_to_remove = random.choice(tmp.edges)

        #     sw_to_connect.add_edge(edg_to_remove.lnode)
        #     sw_to_connect.add_edge(edg_to_remove.rnode)

        #     tmp.remove_edge(edg_to_remove)

        #     self.switches[sw_to_connect.id].add_edge(edg_to_remove.lnode)
        #     self.switches[sw_to_connect.id].add_edge(edg_to_remove.rnode)

        #     if len(self.switches[sw_to_connect.id].edges) > num_ports - 2:
        #         switches_to_connect.remove(sw_to_connect)

        for i in self.switches:
            if len(i.edges) != num_ports:
                print(i)
                print(len(i.edges))

        if plot: self.plot(num_ports)

    def plot(self, num_ports):
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
        plt.savefig("plot_jellyfish_{}.png".format(num_ports))
        plt.clf()

class Fattree:

    def __init__(self, num_ports, plot = False):
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
            fig, ax = plt.subplots(figsize=(num_pods * num_switches_per_layer * PLOT_NODE_SPACING * 2, num_core_switches))
            ax.set_xlim(-1, num_pods * num_switches_per_layer * PLOT_NODE_SPACING)
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


class BCube:

    def __init__(self, num_ports, plot = False):
        self.servers = []
        self.switches = []
        self.generate(num_ports, plot)

    def generate(self, num_ports, plot):

        return


def dijkstra(start_node, switches):
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
                    # save path for reconstruction
                    # path[neighbor] = current_node

        pqueue.task_done()

    for node in unvisited:
        unvisited[node]["path"].append(node)

    return unvisited

def find_path(start_node, end_node, switches):
    distances = dijkstra(start_node, switches)

    if distances[end_node] is None:
        return {"distance": None, "path": []}

    return distances[end_node]


def ksp_yen(start_node, end_node, switches, max_k=2):
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
                distance_total = distances[spur_node]["distance"] + path_spur["distance"]
                potential_path = {"distance": distance_total, "path": path_total}
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
                A.append(next_path)
                break

        # print("k")
    return A