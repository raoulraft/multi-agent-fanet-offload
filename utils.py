import random
import numpy as np
import math
from scipy.io import savemat


def esp_rand(var):
    x = 0
    while x == 0:
        x = np.random.random()
    if var != 0:
        y = -math.log(x) / var
    else:
        y = math.inf

    return y


def esp_rand_zone(var):
    x = 0

    while x == 0:
        x = random.random()
    if var != 0:
        y = -math.log(x) / var
    else:
        y = math.inf

    return y


class Logger:

    def __init__(self) -> object:
        self.delays = []
        self.proc_delays = []
        self.off_delays = []
        self.std_devs = []
        self.off_percentages = []
        self.off_probabilities = []
        self.lost_packets = []
        self.rewards = []
        self.missed_delays = []

    def receive_res(self, results):
        self.rewards.append(results['reward'])
        self.delays.append(results['delay'])
        # print("result:{}".format(results['delay']))
        self.proc_delays.append(results['processing_delay'])
        self.off_delays.append(results['offloading_delay'])
        self.std_devs.append(results['std_dev'])
        self.off_probabilities.append(results['offloading_probability'])
        self.off_percentages.append(results['offloading_percentage'])
        self.lost_packets.append(results['lost_packet'])
        self.missed_delays.append(results['missed_delays'])

    # ignore static method warning.
    # This cannot be static, since we are evaluating objects properties (e.g. self.rewards)
    def write_to_matlab(self, filename):
        results = {}
        results['rewards'] = eval('self.rewards')
        results['delays'] = eval('self.delays')
        results['proc_delays'] = eval("self.proc_delays")
        results['off_delays'] = eval("self.off_delays")
        results['std_devs'] = eval("self.std_devs")
        results['off_percentages'] = eval("self.off_percentages")
        results['off_probabilities'] = eval("self.off_probabilities")
        results['lost_packets'] = eval("self.lost_packets")
        results['missed_delays'] = eval("self.missed_delays")

        savemat(filename, results)
