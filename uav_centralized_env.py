import gym
import numpy as np
import statistics
import wandb
from gym.spaces import Discrete, Box, MultiDiscrete
from drone import Drone, OtherDrone
from zone import Zone
from event import TimeMatrix

INCREASE = [+1, +5, +10, -1, -5, -10, 0]
K = 200


class DronesEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, input_c, result_buffer=None):
        if result_buffer is not None:
            self.save_res = True
            self.res_buffer = result_buffer
        else:
            self.save_res = False
        self.number_of_uavs = input_c.n
        self.processing_rate = input_c.processing_rate
        self.offloading_rate = input_c.offloading_rate
        self.max_time = input_c.max_time

        self.lambdas = [input_c.lmbda_l, input_c.lmbda_h]
        self.prob_trans = input_c.prob_trans

        self.alg = input_c.alg
        self.shifting = input_c.shifting_probs

        self.feature_size = (4 * self.number_of_uavs)  # ProcessingQueue, OffloadingQueue, TrafficPattern, OffProbs
        self.t = 0
        self.tot_reward = 0
        self.obs_max_timer = input_c.obs_max_timer
        self.steps = 0
        if self.alg == "fcto" or self.alg == "woto":
            self.drones = [OtherDrone(self.processing_rate, self.offloading_rate, self.alg) for _ in
                           range(self.number_of_uavs)]
            for drone in self.drones:
                drone.set_drones(self.drones)
        else:
            self.drones = [Drone(self.processing_rate, self.offloading_rate) for _ in range(self.number_of_uavs)]
        self.zones = [Zone(i, self.lambdas[0], self.lambdas[1], i) for i in range(self.number_of_uavs)]
        self.time_matrix = TimeMatrix(self.number_of_uavs, self.prob_trans, self.lambdas)

        self.action_space = MultiDiscrete([7 for _ in range(self.number_of_uavs)]) if self.shifting\
            else MultiDiscrete([10 for _ in range(self.number_of_uavs)])
        self.observation_space = Box(low=0, high=1, shape=(self.feature_size,), dtype=np.float32)

        # tot, proc, ol
        self.avg_tot_delay = [0, 0, 0]
        self.counter_avg_td = [0, 0, 0]
        # computing pdf
        self.arr_delay = [0]
        # counts how many times each zone does a complete low->high->low cycle
        self.count_cycle_zone = np.zeros(self.number_of_uavs)
        # computes mean delay
        self.delay = []
        # computes epoch delay
        self.current_delay = []
        # computes mean offloading probability
        self.offloading_prob = []
        # computes mean reward
        self.mean_reward = []

        # normalize observation between 0 and 1
        self.max_observed_queue = 1
        self.max_observed_queue_ol = 1
        self.max_observed_job_counter = 1

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected

    def render(self, mode="human"):
        pass

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        '''
        if self.alg != "fcto" or self.alg != "woto":
            self.drones = [Drone(self.processing_rate, self.offloading_rate) for _ in range(self.number_of_uavs)]
        else:
            self.drones = [OtherDrone(self.processing_rate, self.offloading_rate, self.alg) for _ in
                           range(self.number_of_uavs)]
            for drone in self.drones:
                drone.set_drones(self.drones)
        self.zones = [Zone(i, self.lambdas[0], self.lambdas[1], i) for i in range(self.number_of_uavs)]
        self.time_matrix = TimeMatrix(self.number_of_uavs, self.prob_trans, self.lambdas)

        # reset metrics
        self.avg_tot_delay = [0, 0, 0]
        self.counter_avg_td = [0, 0, 0]
        self.arr_delay = [0]
        self.count_cycle_zone = np.zeros(self.number_of_uavs)
        self.delay = []
        self.current_delay = []
        self.offloading_prob = []
        self.mean_reward = []
        self.t = 0
        self.steps = 0
        self.tot_reward = 0
        observation = self.get_obs()
        return observation

    def step(self, actions):

        self.steps += 1

        # take action
        if self.shifting:  # increase or decrease current probability
            for i in range(len(self.drones)):
                self.drones[i].change_offloading_probability(INCREASE[actions[i]])
                self.offloading_prob.append(self.drones[i].offloading_prob)
        else:  # set probability to certain value
            for i in range(len(self.drones)):
                self.drones[i].set_offloading_probability(actions[i])
                self.offloading_prob.append(self.drones[i].offloading_prob)

        if self.alg == "ldo":
            for i in range(len(self.drones)):
                self.drones[i].set_offloading_probability(0)
                self.offloading_prob.append(self.drones[i].offloading_prob)

        if self.alg == "us":
            for i in range(len(self.drones)):
                self.drones[i].set_offloading_probability(5)
                self.offloading_prob.append(self.drones[i].offloading_prob)

        [_, _, t_event] = self.time_matrix.search_next_event()

        obs_timer = t_event
        while (t_event - obs_timer) < self.obs_max_timer:
            [row, column, t_event] = self.time_matrix.search_next_event()
            if column == 0:  # CHANGE ACTIVITY STATE
                self.zones[row].change_zone_state(t_event, self.time_matrix)
                if self.zones[row].state == 1:
                    self.count_cycle_zone[row] += 1

            elif column == 1:  # JOB ARRIVAL
                self.drones[self.zones[row].drone_id].job_arrival(row, t_event, self.time_matrix,
                                                                  self.zones)
                self.drones[self.zones[row].drone_id].increase_counter()

                self.update_normalization_counters()

            elif column == 2:  # JOB PROCESSING
                tot_delay, proc_delay, off_delay = self.drones[row].job_processing(row, t_event, self.time_matrix,
                                                                                   self.zones)
                self.update_metrics(tot_delay, proc_delay, off_delay)

            elif column == 3:  # JOB OFFLOADING
                self.drones[row].job_offloading(row, t_event, self.time_matrix, self.zones, self.drones)

            elif column == -1 and row == -1:  # INCREASE HIGH ACTIVITY ZONE LAMDBA
                for i in range(len(self.zones)):
                    self.zones[i].increase_lmbda_high()

            self.t = t_event

        # update metrics (some jobs may be arrived to other queues via offloading event, which doesn't track
        # the receiving drone queues to update the metrics)
        self.update_normalization_counters()

        # retrieve rewards
        reward = - statistics.mean(self.current_delay)
        self.mean_reward.append(reward)
        self.tot_reward += reward
        self.current_delay = []
        # rewards for all agents are placed in the rewards dictionary to be returned
        # rewards = {}
        # rewards = {agent: reward for agent in self.agents}

        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''

        done = self.t >= self.max_time
        # dones = {agent: env_done for agent in self.agents}

        # retrieve observations
        # current observation is just the other player's most recent action
        # observations = {self.agents[i]: int(actions[self.agents[1 - i]]) for i in range(len(self.agents))}
        # obs = self.get_obs()
        # observations = {agent: self.get_obs(aidx) for aidx, agent in enumerate(self.agents)}
        observation = self.get_obs()
        for drone in self.drones:
            drone.clear_buffer()

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        # infos = {agent: {} for agent in self.agents}
        info = {}
        if done:
            mean_delay = statistics.mean(self.delay)
            mean_reward = statistics.mean(self.mean_reward)
            jitter = statistics.stdev(self.delay)
            off_probs = [drone.offloading_prob for drone in self.drones]
            mean_off = statistics.mean(self.offloading_prob)
            max_q = max([drone.max_queue_length for drone in self.drones])
            max_q_o = max([drone.max_ol_queue_length for drone in self.drones])
            mean_q = statistics.mean([drone.get_mean_queue()[0] for drone in self.drones])
            mean_q_o = statistics.mean([drone.get_mean_queue()[1] for drone in self.drones])
            lost_p = sum([drone.lost_pkts for drone in self.drones])
            arrived_p = sum([drone.arrived_pkts for drone in self.drones])
            lost_percentage = lost_p / arrived_p
            wandb.log({"episode reward": self.tot_reward}, commit=False)
            wandb.log({"mean reward": mean_reward}, commit=False)
            wandb.log({"lost packet percentage": lost_percentage}, commit=False)
            wandb.log({"max processing queue": max_q}, commit=False)
            wandb.log({"max offloading queue": max_q_o}, commit=False)
            wandb.log({"mean processing queue": mean_q}, commit=False)
            wandb.log({"mean offloading queue": mean_q_o}, commit=False)
            wandb.log({"mean offloading probabilities": mean_off}, commit=False)
            wandb.log({f"final offloading probability - {d_idx}": drone.offloading_prob
                       for d_idx, drone in enumerate(self.drones)}, commit=False)
            wandb.log({"jitter": jitter}, commit=False)

            wandb.log({"episode mean delay": mean_delay}, commit=True)
            # if self.save_res:
            #     self.res_buffer.save_run_results(avg_delay=mean_delay, jitter=jitter, reward=self.tot_reward,
            #                                  offloading_ratio=mean_off, lost_jobs=lost_percentage)

        return observation, reward, done, info

    def get_obs(self):

        out = np.full((self.feature_size), 0.0)

        for i in range(len(self.drones)):
            out[i] = self.drones[i].queue / self.max_observed_queue

        for i in range(len(self.drones)):
            out[i + len(self.drones)] = self.drones[i].queue_ol / self.max_observed_queue_ol

        for i in range(len(self.drones)):
            out[i + 2 * len(self.drones)] = self.drones[i].job_counter_obs / self.max_observed_job_counter

        for i in range(len(self.drones)):
            out [i + 3 * len(self.drones)] = self.drones[i].offloading_prob / 100

        out = np.array(out)

        return out

    def update_normalization_counters(self):
        for i in range(len(self.drones)):
            if self.drones[i].queue > self.max_observed_queue:
                self.max_observed_queue = self.drones[i].queue
            if self.drones[i].queue_ol > self.max_observed_queue_ol:
                self.max_observed_queue_ol = self.drones[i].queue_ol
            if self.drones[i].job_counter_obs > self.max_observed_job_counter:
                self.max_observed_job_counter = self.drones[i].job_counter_obs

    def update_metrics(self, tot_delay, proc_delay, off_delay):

        self.delay.append(tot_delay)  # former delay_arr
        self.current_delay.append(tot_delay)  # to calculate mean delay of the whole network during epoch

        # need the try catch block since the max lenghts of the array (K/mu + Kol/muol)
        # is not the real possible max delay obtainable (because of esp_rand function)
        try:
            self.arr_delay[int(tot_delay)] += 1
        except:
            assert int(tot_delay) >= 0
            self.arr_delay.extend(((int(tot_delay) + 1) - len(self.arr_delay)) * [0])
            self.arr_delay[(int(tot_delay))] = 1

        delay = np.zeros(3)
        delay[0] = tot_delay
        delay[1] = proc_delay
        off = False
        if off_delay is not None:
            delay[2] = off_delay
            off = True
        for i in range(len(self.avg_tot_delay)):
            if i != 2 or off:
                self.counter_avg_td[i] += 1
                counter = self.counter_avg_td[i]
                avg_delay = self.avg_tot_delay[i]
                new_delay = delay[i]
                self.avg_tot_delay[i] = (((counter - 1) * avg_delay) + new_delay) / counter