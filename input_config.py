import numpy as np

K = 150
K_ol = 150


class InputConfig:
    def __init__(self, uavs, processing_rate, offloading_rate, lmbda, prob_trans, shifting_probs, algorithm,
                 max_time=10000,
                 obs_time=10,
                 frame_stack=4, ):
        self.n = uavs
        self.alg = algorithm
        self.frame_stack = frame_stack
        self.processing_rate = processing_rate
        self.offloading_rate = offloading_rate
        self.max_time = max_time
        self.lmbda_l = lmbda[0]
        self.lmbda_h = lmbda[1]
        self.prob_trans = prob_trans
        self.shifting_probs = shifting_probs
        P = np.array([[(1 - prob_trans[0]), prob_trans[0]], [prob_trans[1], (1 - prob_trans[1])]])
        Aa = np.append(np.transpose(P) - np.identity(2), [[1, 1]], axis=0)
        bB = np.transpose(np.array([0, 0, 1]))
        self.avg_residence_time = np.linalg.solve(np.transpose(Aa).dot(Aa), np.transpose(Aa).dot(bB))
        self.lambda_tot = (self.lmbda_l * self.avg_residence_time[0]) + (self.lmbda_h * self.avg_residence_time[1])
        self.p_tot = self.lambda_tot / self.processing_rate
        self.obs_max_timer = obs_time

    def print_settings(self):
        print("Number of Uavs:", self.n)
        print("Processing rate:", self.processing_rate)
        print("Offloading rate:", self.offloading_rate)
        print("Lambda {} | {}".format(self.lmbda_l, self.lmbda_h))
        print("Total lambda:", self.lambda_tot)
        print("Prob trans {} | {}".format(self.prob_trans[0], self.prob_trans[1]))
        print("ro {} | {}".format(self.lmbda_l / self.processing_rate, self.lmbda_h / self.processing_rate))
        print("Total ro:", self.p_tot)
