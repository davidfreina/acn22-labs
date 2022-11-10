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

		edge_cnt, switch_id = 0

		for server in range(num_servers):
			num_connections = random.randrange(max_server_per_switch)

			switch_node = Node(switch_id, "tor_switch")

			for edg in range(num_connections):
				server_node = Node(server + edg + edge_cnt, "server")
				switch_node.add_edge(server_node)
				self.servers[server + edg + edge_cnt] = server_node

			self.switches[switch_id] = switch_node

			switch_id += 1
			edge_cnt += num_connections

			if(edge_cnt == num_servers):
				while(switch_id < num_switches):
					self.switches[switch_id] = Node(switch_id, "net_switch")
					switch_id += 1
				break




		while(True):







class Fattree:

	def __init__(self, num_ports):
		self.servers = []
		self.switches = []
		self.generate(num_ports)

	def generate(self, num_ports):
		# TODO: code for generating the fat-tree topology
		num_pods = num_ports
		num_hosts = (num_pods ** 3) / 4
		num_switches_per_layer = num_pods // 2
		num_core_switches = (num_pods // 2) ** 2

		# core_switches = {}

		pod_switches = {}

		for sw_core_row_idx in range(0, num_core_switches // num_switches_per_layer):
			core_switches = []
			for sw_core_column_idx in range(0, num_switches_per_layer):
				core_sw_id = "10.{}.{}.{}".format(num_pods, sw_core_row_idx + 1, sw_core_column_idx + 1)
				core_sw = Node(core_sw_id, "core-sw")
				core_switches.append(core_sw)
				self.switches[core_sw_id] = core_sw

			for pod_idx in range(0, num_pods):
				pod_switches.setdefault(pod_idx, {"agg-sw": [], "edge-sw": []})

				agg_sw_id = "10.{}.{}.{}".format(pod_idx, num_switches_per_layer + sw_core_row_idx, 1)
				agg_sw = Node(agg_sw_id, "agg-sw")
				for core_sw in core_switches:
					core_sw.add_edge(agg_sw)
					# agg_sw.add_edge(core_sw)
				pod_switches[pod_idx]["agg-sw"].append(agg_sw)
				self.switches[agg_sw_id] = agg_sw

				edge_sw_id = "10.{}.{}.{}".format(pod_idx, sw_core_row_idx, 1)
				edge_sw = Node(edge_sw_id, "edge-sw")
				# agg_sw.add_edge(edge_sw)
				# edge_sw.add_edge(agg_sw)
				pod_switches[pod_idx]["edge-sw"].append(edge_sw)
				self.switches[edge_sw_id] = edge_sw

				for srv_idx in range(1, (num_pods // 2) + 1):
					srv_id = "10.{}.{}.{}".format(pod_idx, sw_core_row_idx, srv_idx + 1)
					srv = Node(srv_id, "server")
					self.servers[srv_id] = srv
					edge_sw.add_edge(srv)


		for pod in pod_switches:
			for agg_sw in pod_switches[pod]["agg-sw"]:
				for edge_sw in pod_switches[pod]["edge-sw"]:
					agg_sw.add_edge(edge_sw)
					# edge_sw.add_edge(agg_sw)

			# ID of core switch definition:
			# 10.k.j.i, where j and i denote that switches
			# coordinates in the (k/2)^2 core switch grid
			# (each in [1, (k/2)], starting from top-left).
			# calculate j:
			# sw_core_row = sw_core_idx // num_switches_per_layer + 1
			# calculate i:
			# sw_core_column = sw_core_idx % num_switches_per_layer + 1

			# create core switch id and add it to dictionary
			# with array containing the pod switch id's for each core sw
			# core_sw_id = "10.{}.{}.{}".format(num_pods, sw_core_row, sw_core_column)
			# core_sw = Node(core_sw_id, "core-sw")
#			core_switches.setdefault(core_sw_id, set())

			# iterate over pods and add pod sw id's to core sw array
			# also add pod sw id to dictionary with pod index as key
			# for pod_idx in range(0, num_pods):
			# 	pod_sw_id = "10.{}.{}.{}".format(pod_idx, num_switches_per_layer + sw_core_row - 1, 1)
			# 	core_switches[core_sw_id].add(pod_sw_id)

			# print(core_sw_id)

		# for pod_idx in range(0, num_pods):
		# 	# print("POD {}".format(pod_idx))
		# 	for sw_pod_idx in range(0, num_pods):
		# 		pod_sw_id = "10.{}.{}.{}".format(pod_idx, sw_pod_idx, 1)
		# 		pod_sw = Node(core_sw_id, "pod-sw")
		# 		# pod_switches.setdefault(pod_sw_id, set())
		# 		for sw_pod_link in range(0, num_switches_per_layer):
		# 			if sw_pod_idx // num_switches_per_layer == 0:
		# 				pod_sw_link = "10.{}.{}.{}".format(pod_idx, num_switches_per_layer + sw_pod_link, 1)
		# 			else:
		# 				pod_sw_link = "10.{}.{}.{}".format(pod_idx, sw_pod_link, 1)
		# 			pod_switches[pod_sw_id].add(pod_sw_link)
				# print(pod_sw_id)

		# print(core_switches)
		# print(pod_switches)

		# self.switches = {**core_switches, **pod_switches}

		for switch in switches:
			print(switch)