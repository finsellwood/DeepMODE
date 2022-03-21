import numpy as np
import pickle
rootpath = "/vols/cms/fjo18/Masters2021/Nathan/"


class NetworkBA:
    def __init__(self, m, max_time_step, n0, attachment=1, r=0):
        self.m = m
        self.r = r
        self.n0 = n0
        self.max_time_step = max_time_step  # run time increases propto N**2    40000 => 100s
        self.attachment = attachment
        self.arr = np.zeros((n0, n0), dtype=int) - np.identity(n0, dtype=int)
        self.network = []
        self.k_list = []

    def initial_net(self):
        for step in range(0, self.m * self.n0):
            idx_arr = np.argwhere(self.arr == 0)
            random_idx = np.random.randint(len(idx_arr))
            i, j = idx_arr[random_idx]
            self.arr[i][j] = 1
            self.arr[j][i] = 1

        for step in range(0, self.n0):
            idx_arr = np.where(self.arr[:][step] == 1)
            if len(idx_arr[0]) == 0:
                idx_arr = [step, np.random.choice(np.delete(np.arange(self.n0), step), size=1)[0]]
                print("unconnected node")
                print(idx_arr)
                self.network.append([idx_arr[1]])
                self.k_list.append(1)
            else:
                self.network.append(list(idx_arr[0]))
                self.k_list.append(len(idx_arr[0]))

        self.network = list(self.network)

        """
        self.n0 = self.m+1
        networknew = []
        for node in range(0, self.n0):
            networknew.append(list(np.delete(np.arange(0, self.n0), node)))
        self.k_list = list(np.zeros(self.n0, dtype=int) + self.m)
        
        self.network = networknew
        """
        print("Initial network created")

    def run(self):
        self.initial_net()
        t = 0
        while t < self.max_time_step:
            indx_list = np.arange(self.n0 + t)
            if (t%5000) == 0:
                print("progress report: " + str(100*t/self.max_time_step) + "%")
            if self.attachment == 1:  # preferential picks
                picked = np.random.choice(indx_list, size=self.m - self.r, replace=False,
                                          p=np.array(self.k_list) / sum(self.k_list))
                # print('preferential selected')
            else:  # random picks
                picked = np.random.choice(indx_list, size=self.m - self.r, replace=False)
                # print('random selected')

            k_list_copy = np.array(self.k_list.copy(), dtype=int)
            for step in range(0, self.r):
                i = int(np.random.choice(indx_list, size=1, replace=False,
                                         p=k_list_copy / np.sum(k_list_copy))[0])
                unconnected_nodes = list(set(indx_list).difference(self.network[i] + [i]))
                j = int(np.random.choice(unconnected_nodes, size=1, replace=False,
                                         p=k_list_copy[unconnected_nodes] / np.sum(
                                             k_list_copy[unconnected_nodes]))[0])
                self.network[i].append(j)
                self.network[j].append(i)
                self.k_list[i] += 1
                self.k_list[j] += 1

            for pick in picked:
                self.k_list[pick] += 1
                self.network[pick].append(self.n0 + t)

            self.network.append(list(picked))
            self.k_list.append(self.m - self.r)
            t += 1

    def save(self, type, iteration, type_attach):
        if type == "k":
            data = self.k_list
        elif type == "N":
            data = self.network
        else:
            raise Exception("Specified data structure not recognised")

        filename = rootpath + str(type) + '_data_' + type_attach + '_N' + str(self.max_time_step) \
                   + '_' + str(self.m) + '_' + str(iteration) + '.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def print_state(self):
        print(self.k_list)
        print(self.network)
