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

import matplotlib.pyplot as plt
import sys
from operator import itemgetter
import itertools
import random
from pathos.multiprocessing import Pool
import os
import topo
sys.setrecursionlimit(2147483647)

# Setup for Jellyfish
num_servers = 686
num_switches = 245
num_ports = 14

jf_topo = topo.Jellyfish(num_servers, num_switches, num_ports)

# TODO: code for reproducing Figure 9 in the jellyfish paper


def calculate_edges(links):
    edges = {}
    for start_node in links:
        paths = links[start_node]
        for path_info in paths:
            path = path_info["path"]
            # print(path)
            for path_node_index in range(0, len(path) - 1):
                current_node = path[path_node_index]
                next_node = path[path_node_index + 1]
                edge = current_node.get_edge(next_node)

                existing_edge = [x for x in edges if ((x.rnode.id == edge.rnode.id and x.lnode.id == edge.lnode.id) or (
                    x.rnode.id == edge.lnode.id and x.lnode.id == edge.rnode.id))]

                if not existing_edge:
                    edges.setdefault(edge, 0)
                    existing_edge.append(edge)
                edges[existing_edge[0]] += 1
    return edges


def calculate_plot_input(edges):
    ret = [[0], [0]]
    sortedDict = sorted(edges.items(), key=itemgetter(1))

    lastRank = 0
    lastVal = 0
    for pair in sortedDict:
        lastRank += 1
        if pair[1] != lastVal:
            ret[0].append(lastRank)
            ret[1].append(lastVal)
            lastVal = pair[1]
    return ret


jf_servers = jf_topo.servers.copy()
random.shuffle(jf_servers)

links = {}
# start = time()

permutations = list(zip(jf_topo.servers, jf_servers, itertools.repeat(
    (jf_topo.switches + jf_topo.servers)), itertools.repeat(64)))


with Pool(processes=os.cpu_count()) as pool:
    results = pool.starmap(topo.ksp_yen, permutations)

for paths in results:
    links[paths[0]["path"][0]] = paths

k_8 = {}
ecmp_8 = {}
ecmp_64 = {}

for start_node in links:
    k_8.setdefault(start_node, [])
    ecmp_8.setdefault(start_node, [])
    ecmp_64.setdefault(start_node, [])

    paths = sorted(links[start_node], key=itemgetter("distance"))

    k_8[start_node].append(paths[0])
    ecmp_8[start_node].append(paths[0])
    ecmp_64[start_node].append(paths[0])

    for i in range(1, min(8, len(paths))):
        k_8[start_node].append(paths[i])
        if ecmp_8[start_node][-1]["distance"] == paths[i]["distance"]:
            ecmp_8[start_node].append(paths[i])

    for i in range(1, min(64, len(paths))):
        if ecmp_64[start_node][-1]["distance"] == paths[i]["distance"]:
            ecmp_64[start_node].append(paths[i])
        else:
            break

# print("parallel: {}s".format(time() - start))
# start = time()

edges_k8 = calculate_edges(k_8)
edges_ecmp8 = calculate_edges(ecmp_8)
edges_ecmp64 = calculate_edges(ecmp_64)


ret_k8 = calculate_plot_input(edges_k8)
ret_ecmp8 = calculate_plot_input(edges_ecmp8)
ret_ecmp64 = calculate_plot_input(edges_ecmp64)

plt.switch_backend('Agg')
plt.step(ret_k8[0], ret_k8[1], label='8 Shortest Paths')
plt.step(ret_ecmp8[0], ret_ecmp8[1], label='8-way ECMP')
plt.step(ret_ecmp64[0], ret_ecmp64[1], label='64-way ECMP')
plt.legend()
plt.margins(x=0)
plt.margins(y=0)
plt.xlabel("Rank of Link")
plt.ylabel("# of distinct paths link is on")
plt.savefig('9.png')
plt.close()

# print("Yens: {}s".format(end - start))
