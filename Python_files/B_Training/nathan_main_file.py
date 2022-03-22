# This is the main script

import NetworksOOP as NetO
import timeit
import pickle
import numpy as np

#N_list = [1000, 10000, 100000, 1000000]  # 100, 500, 1000, 5000, 10000, 50000, 100000,
N_list = [100000]
m_list = [2,4,8,16,32]
initial_nodes = 75
no_iterations = 2000000

offset = 80
for N in N_list:
    for iteration in range(offset, offset + int(no_iterations / N)):
        for m in m_list:
            Nint = int(N)
            print(Nint, m, int(m/2))
            starttime = timeit.default_timer()

            network = NetO.NetworkBA(m, Nint, initial_nodes, r=int(m/2))  # m, max_time_step, n0, attachment=1, r=0
            network.run()
            #network.print_state()
            mix = 'mix'+str(m)
            network.save("k", iteration, mix)  # type, iteration
            #network.save("N", iteration, "ra")  # type, iteration

            print("The time difference is :", timeit.default_timer() - starttime)
