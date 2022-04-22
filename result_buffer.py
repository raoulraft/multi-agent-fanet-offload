import numpy as np
import statistics
from random import random
from scipy.io import savemat
import os


class ResultBuffer:

    def __init__(self, min_n_drone, max_n_drone, min_mu, max_mu, step_mu, net_slice, change_processing, alg):
        # TODO: add run id for saving the csv file to that directory (and avoid overwriting old runs)
        self.id = random()
        self.dir = "./csv/"
        self.var = "p" if change_processing else "o"  # for the filename
        os.makedirs(self.dir, exist_ok=True)
        self.len_n = int(max_n_drone - min_n_drone) + 1  # 9 - 4 = 5  4->0 5->1 6->2 7->3 8->4 9->5
        self.net_slice = net_slice  # network slice
        self.max_n = max_n_drone
        self.min_n = min_n_drone
        self.len_mu = int((max_mu - min_mu) / step_mu) + 1  # (3.15 - 2.15) / step_mu_p
        self.step_mu = step_mu  # e.g. runs from 2.1 to 3.1 with 0.1 step
        self.max_mu = max_mu
        self.min_mu = min_mu
        self.alg = alg

        """
        self.delay = np.zeros((self.len_n, self.len_mu), )  # delay final results
        self.jitter = np.zeros((self.len_n, self.len_mu), )  # jitter final results
        self.reward = np.zeros((self.len_n, self.len_mu), )  # reward final results
        self.offloading_ratio = np.zeros((self.len_n, self.len_mu), )  # offloading ratio final results
        self.lost_jobs = np.zeros((self.len_n, self.len_mu), )
        """
        self.delay = np.zeros(self.len_mu)  # delay final results
        self.jitter = np.zeros(self.len_mu)  # jitter final results
        self.reward = np.zeros(self.len_mu)  # reward final results
        self.offloading_ratio = np.zeros(self.len_mu) # offloading ratio final results
        self.lost_jobs = np.zeros(self.len_mu)
        # arrays to save several runs results.
        # these will be processed and then discarded (reset to empty array) after the run is finished
        self.current_delay = []
        self.current_jitter = []
        self.current_reward = []
        self.current_offloading = []
        self.current_lost = []

        # saving temp results into csv file
        d = np.asarray(self.delay)
        j = np.asarray(self.jitter)
        r = np.asarray(self.reward)
        o = np.asarray(self.offloading_ratio)
        l = np.asarray(self.lost_jobs)

        print(f"initializing buffer with id {self.id}")

    def set_save_runs(self, n_drones, mu):
        self.n_drones_save = n_drones
        self.mu_save = mu

    # not used (doesn't work in parallel envs / multi agent envs)
    def save_and_reset(self, n_drones, mu):
        n_drones_idx = int(n_drones - self.min_n)
        print("mu", mu, " - min_mu", self.min_mu, "/ step mu", self.step_mu)
        x = (mu - self.min_mu) / self.step_mu
        print("res:", x)
        print("int res:", int(x))
        mu_p_idx = int((mu - self.min_mu) / self.step_mu)
        print("index:", n_drones_idx, mu_p_idx)

        print(f"[{self.id}] \n {self.current_delay}")
        """
        self.delay[n_drones_idx, mu_p_idx] = statistics.mean(self.current_delay)
        self.jitter[n_drones_idx, mu_p_idx] = statistics.mean(self.current_jitter)
        self.reward[n_drones_idx, mu_p_idx] = statistics.mean(self.current_reward)
        self.offloading_ratio[n_drones_idx, mu_p_idx] = statistics.mean(self.current_offloading)
        self.lost_jobs[n_drones_idx, mu_p_idx] = statistics.mean(self.current_lost)
        """
        self.delay[n_drones_idx, mu_p_idx] = statistics.mean(self.current_delay)
        self.jitter[n_drones_idx, mu_p_idx] = statistics.mean(self.current_jitter)
        self.reward[n_drones_idx, mu_p_idx] = statistics.mean(self.current_reward)
        self.offloading_ratio[n_drones_idx, mu_p_idx] = statistics.mean(self.current_offloading)
        self.lost_jobs[n_drones_idx, mu_p_idx] = statistics.mean(self.current_lost)
        # saving temp results into csv file
        d = np.asarray(self.delay)
        j = np.asarray(self.jitter)
        r = np.asarray(self.reward)
        o = np.asarray(self.offloading_ratio)
        l = np.asarray(self.lost_jobs)

        np.savetxt(self.dir + f"delay{self.net_slice}{self.var}.csv", d, delimiter=",")
        np.savetxt(self.dir + f"jitter{self.net_slice}{self.var}.csv", j, delimiter=",")
        np.savetxt(self.dir + f"reward{self.net_slice}{self.var}.csv", r, delimiter=",")
        np.savetxt(self.dir + f"offload_ratio{self.net_slice}{self.var}.csv", o, delimiter=",")
        np.savetxt(self.dir + f"lost_jobs{self.net_slice}{self.var}.csv", l, delimiter=",")

        # resetting arrays
        self.current_delay = []
        self.current_jitter = []
        self.current_reward = []
        self.current_offloading = []
        self.current_lost = []

    def save_run_results(self, avg_delay, jitter, reward, offloading_ratio, lost_jobs):
        try:
            self.delay = np.loadtxt(self.dir + f"delay{self.n_drones_save}{self.var}{self.alg}.csv", delimiter=',')
            self.jitter = np.loadtxt(self.dir + f"jitter{self.n_drones_save}{self.var}{self.alg}.csv", delimiter=",")
            self.reward = np.loadtxt(self.dir + f"reward{self.n_drones_save}{self.var}{self.alg}.csv", delimiter=",")
            self.offloading_ratio = np.loadtxt(self.dir + f"offload_ratio{self.n_drones_save}{self.var}{self.alg}.csv", delimiter=",")
            self.lost_jobs = np.loadtxt(self.dir + f"lost_jobs{self.n_drones_save}{self.var}{self.alg}.csv", delimiter=",")
        except OSError:  # if csv is not present, then it must be initialized for the first time. Hence, vars are set
            # to np.zeros
            self.delay = np.zeros(self.len_mu)  # delay final results
            self.jitter = np.zeros(self.len_mu)  # jitter final results
            self.reward = np.zeros(self.len_mu)  # reward final results
            self.offloading_ratio = np.zeros(self.len_mu)  # offloading ratio final results
            self.lost_jobs = np.zeros(self.len_mu)

        # print(self.delay)
        self.current_delay.append(avg_delay)
        self.current_jitter.append(jitter)
        self.current_reward.append(reward)
        self.current_offloading.append(offloading_ratio)
        self.current_lost.append(lost_jobs)



        # print(f"[{self.id}] \n {self.current_delay}")

        # n_drones_idx = self.n_drones_save
        # mu_p_idx = self.mu_save

        n_drones_idx = int(self.n_drones_save - self.min_n)
        # print("mu", self.mu_save, " - min_mu", self.min_mu, "/ step mu", self.step_mu)
        x = (self.mu_save - self.min_mu) / self.step_mu
        # print("res:", x)
        # print("int res:", int(x))
        mu_p_idx = int((self.mu_save - self.min_mu) / self.step_mu)
        # print("index:", n_drones_idx, mu_p_idx)

        """
        self.delay[n_drones_idx, mu_p_idx] = statistics.mean(self.current_delay)
        self.jitter[n_drones_idx, mu_p_idx] = statistics.mean(self.current_jitter)
        self.reward[n_drones_idx, mu_p_idx] = statistics.mean(self.current_reward)
        self.offloading_ratio[n_drones_idx, mu_p_idx] = statistics.mean(self.current_offloading)
        self.lost_jobs[n_drones_idx, mu_p_idx] = statistics.mean(self.current_lost)
        """
        self.delay[mu_p_idx] = statistics.mean(self.current_delay)
        self.jitter[mu_p_idx] = statistics.mean(self.current_jitter)
        self.reward[mu_p_idx] = statistics.mean(self.current_reward)
        self.offloading_ratio[mu_p_idx] = statistics.mean(self.current_offloading)
        self.lost_jobs[mu_p_idx] = statistics.mean(self.current_lost)

        # saving temp results into csv file
        d = np.asarray(self.delay)
        j = np.asarray(self.jitter)
        r = np.asarray(self.reward)
        o = np.asarray(self.offloading_ratio)
        l = np.asarray(self.lost_jobs)

        np.savetxt(self.dir + f"delay{self.n_drones_save}{self.var}{self.alg}.csv", d, delimiter=",")
        np.savetxt(self.dir + f"jitter{self.n_drones_save}{self.var}{self.alg}.csv", j, delimiter=",")
        np.savetxt(self.dir + f"reward{self.n_drones_save}{self.var}{self.alg}.csv", r, delimiter=",")
        np.savetxt(self.dir + f"offload_ratio{self.n_drones_save}{self.var}{self.alg}.csv", o, delimiter=",")
        np.savetxt(self.dir + f"lost_jobs{self.n_drones_save}{self.var}{self.alg}.csv", l, delimiter=",")
