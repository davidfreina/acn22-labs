import sys

import topo

if __name__ == "__main__":
    # Same setup for Jellyfish and Fattree
    # called with number of ports
    try:
        num_ports = int(sys.argv[1])
    except IndexError:
        num_ports = 4

    num_servers = int(num_ports ** 3 / 4)
    # ((num_ports / 2) ** 2) + (num_ports ** num_ports) -> core switches + agg. and edge switches
    # formula simplified
    num_switches = int(5 * num_ports ** 2 / 4)

    print("Generating Topology for: {} ports, {} servers and {} switches".format(
        num_ports, num_servers, num_switches))

    ft_topo = topo.Fattree(num_ports, True)
    jf_topo = topo.Jellyfish(num_servers, num_switches, num_ports, True)
