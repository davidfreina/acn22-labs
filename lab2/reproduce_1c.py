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

from time import time
import matplotlib.pyplot as plt
import numpy as np
import topo

# if __name__ == "__main__":
#     print("main")

# Same setup for Jellyfish and Fattree
num_servers = 686
num_switches = 245
num_ports = 14

start = time()
ft_topo = topo.Fattree(num_ports)
end = time()

print("Fattree topo took {}".format(end - start))

# TODO: code for reproducing Figure 1(c) in the jellyfish paper

ft_distances = {}
# use array from 0..6 for fraction calculation
ft_path_lengths = [0] * 7

ft_switches = ft_topo.switches.copy()

start = time()
for src_sw_srv in ft_topo.switches:
    distances = topo.dijkstra(src_sw_srv, ft_switches)
    for dest_sw_srv in distances:
        ft_distances[(src_sw_srv.id, dest_sw_srv.id)] = distances[dest_sw_srv]["distance"]
        # For edge switch add the reverse route as well
        if src_sw_srv.type == "edge-sw":
            ft_distances[(dest_sw_srv.id, src_sw_srv.id)] = distances[dest_sw_srv]["distance"]
    # remove edge switch because it is not needed anymore
    # significant speed up achieved
    if src_sw_srv.type == "edge-sw":
        ft_switches.remove(src_sw_srv)


ft_edge_switches = [server.edges[0].lnode.id for server in ft_topo.servers]

for idx, curr_edge_sw in enumerate(ft_edge_switches):
    for other_edge_sw in ft_edge_switches[idx + 1:]:
        # Add 2 for connection from host -- path -- host
        host_host_distance = ft_distances[(curr_edge_sw, other_edge_sw)] + 2
        ft_path_lengths[host_host_distance] += 1


ft_path_length_distribution = [length / sum(ft_path_lengths[2:7]) for length in ft_path_lengths[2:7]]
end = time()
print("Fattree dijkstra took {}".format(end - start))



# use array from 0..6 for fraction calculation
jf_path_lengths = [0] * 7

for i in range(1):
    start = time()
    jf_topo = topo.Jellyfish(num_servers, num_switches, num_ports)
    end = time()

    print("Jellyfish topo #{}  took {}".format(i, end - start))

    jf_distances = {}

    start = time()
    for src_sw_srv in jf_topo.switches:
        distances = topo.dijkstra(src_sw_srv, jf_topo.switches)
        for dest_sw_srv in distances:
            jf_distances[(src_sw_srv.id, dest_sw_srv.id)] = distances[dest_sw_srv]["distance"]
            # jf_path_lengths[distances[dest_sw_srv]] += 1

    jf_switches = [server.edges[0].lnode.id for server in jf_topo.servers]
    for idx, curr_edge_sw in enumerate(jf_switches):
        for other_edge_sw in jf_switches[idx + 1:]:
            # Add 2 for connection from host -- path -- host
            host_host_distance = jf_distances[(curr_edge_sw, other_edge_sw)] + 2
            jf_path_lengths[host_host_distance] += 1
    end = time()
    print("Jellyfish dijkstra #{}  took {}".format(i, end - start))

jf_path_length_distribution = [length / sum(jf_path_lengths[2:7]) for length in jf_path_lengths[2:7]]




# See: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
labels = ['2', '3', '4', '5', '6']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x + width/2, ft_path_length_distribution, width, label="Fat-tree")
rects2 = ax.bar(x - width/2, jf_path_length_distribution, width, label="Jellyfish")

plt.ylim([0, 1])
ax.set_ylabel("Path length")
ax.set_title("Path length distribution for num_ports={}".format(num_ports))
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()
fig.savefig("1c.png")