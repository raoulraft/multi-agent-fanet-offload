from packet import Packet
from input_config import K, K_ol
import math
import random


class Drone:
    def __init__(self, processing_rate, offloading_rate):
        # all the counters are used to easily calculate the mean of some metrics without memory consumption
        # avg_metric_in_t1= ((counter-1) * avg_metric_in_t0 + new_value_in_t1) / counter
        # (counter increases each time a new_value has to be included in the mean)

        self.queue = 0  # processing queue counter (== len(o_queue))
        self.queue_ol = 0  # offloading queue counter (== len(p_queue))
        self.p_queue = []  # Packet() processing queues
        self.o_queue = []  # Packet() offloading queues

        self.processing_rate = processing_rate  # CPU processing rate

        self.offloading_rate = offloading_rate  # Offloading rate (depends on link and job size)
        self.offloading_prob = 0  # offloading probability currently used

        self.lost_pkts = 0  # USED by uav_env to calculate lost percentage
        self.arrived_pkts = 0  # USED by uav_env by uav_env to calculate lost percentage
        self.mean_queue_length = [0, 0]  # USED by uav_env to calculate mean queues: calculate mean processing [0]
        # and offloading [1] queues
        self.mean_queue_length_counter = [0, 0]  # counters for calculating mean processing and offloading queues

        self.processed_pkts = 0  # can be useful to get offloading percentages
        self.offloaded_pkts = 0  # can be useful to get offloading percentages

        self.running_delay = [0, 0, 0, 0]  # USED by uav_env to calculate reward based on this epoch mean delay:
        # [mean episode delay, counter for mean episode delay, mean epoch delay, counter for mean epoch delay]

        self.job_counter_obs = 0  # USED to get a normalized observation of the jobs arrived in the previous epoch (
        # number of jobs arrived in previous epoch, to understand if the zone is in high or low activity)
        self.processed_job_counter_obs = 0  # COULD be used to get a normalized observation of the number of
        # processed jobs in the previous epoch (maybe to understand processing rate, but it could be useless)

        self.empty_queue_timer = 0  # COULD be used to track the time spent with an empty processing queue (optional)
        self.start_timer = 0  # COULD be used to get empty queue probability (at the end of the run) (optional)

        self.max_queue_length = 0  # USED by uav_env to get the max processing queue value in that episode (optional)
        self.max_ol_queue_length = 0  # USED by uav_env to get the max offloading queue value in that episode (optional)

        self.alg = "default"  # default when using offloading probabilities. Other values are used to implement
        # different algorithms

    # Called each time a job arrives from the ground: decides whether the job will be offloaded or not. If the job
    # gets offloaded, it will be enqueues in the offloading queue, otherwise it will be enqueued in the processing
    # queue
    def job_arrival(self, row, t_event, time_matrix, zones, offloaded_packet=None):
        assert self.queue <= K
        assert self.queue_ol <= K_ol
        self.increase_counter()
        self.arrived_pkts += 1
        if offloaded_packet is None:  # packet generated from zone
            # check offload probability and decide where to enqueue the packet
            decision = random.uniform(0, 1)
            if decision > (self.offloading_prob / 100):  # local processing

                if self.queue < K:
                    self.p_queue.append(Packet(t_event))
                    self.queue += 1

                    if self.queue == 0:  # stop timer for increasing empty queue Computing Element probability
                        self.empty_queue_timer += t_event - self.start_timer

                    if self.queue > self.max_queue_length:
                        self.max_queue_length = self.queue

                    self.mean_queue_length_counter[0] += 1
                    self.mean_queue_length[0] = (((self.mean_queue_length_counter[0] - 1) * self.mean_queue_length[0])
                                                 + self.queue) / self.mean_queue_length_counter[0]
                else:
                    self.lost_pkts += 1
                if time_matrix.matrix[zones[row].drone_id][2] == math.inf:  # if the processing queue is empty,
                    # immediately start to process the job
                    time_matrix.update_matrix(self.processing_rate, zones[row].drone_id, 2, t_event)
            else:  # offloading
                if self.queue_ol < K_ol:
                    self.o_queue.append(Packet(t_event))
                    self.queue_ol += 1
                    if self.queue_ol > self.max_ol_queue_length:
                        self.max_ol_queue_length = self.queue_ol
                    self.mean_queue_length_counter[1] += 1
                    self.mean_queue_length[1] = (((self.mean_queue_length_counter[1] - 1) * self.mean_queue_length[1])
                                                 + self.queue) / self.mean_queue_length_counter[1]
                else:
                    self.lost_pkts += 1
                if time_matrix.matrix[zones[row].drone_id][3] == math.inf:  # if the offloading queue is empty,
                    # immediately trigger an offloading event
                    time_matrix.update_matrix(self.offloading_rate, zones[row].drone_id, 3, t_event)

            zones[row].schedule_next_arrival(time_matrix, t_event)
            # schedules the next job arrival (this happens only if the packet was not offloaded )

        else:  # packet needs to be processed locally, and delay needs to be retrieved
            if self.queue < K:
                if self.queue == 0:  # stop timer for increasing empty queue CE probability
                    self.empty_queue_timer += t_event - self.start_timer
                self.p_queue.append(offloaded_packet)
                self.queue += 1
                self.mean_queue_length_counter[0] += 1
                self.mean_queue_length[0] = (((self.mean_queue_length_counter[0] - 1) * self.mean_queue_length[0])
                                             + self.queue) / self.mean_queue_length_counter[0]
            else:
                self.lost_pkts += 1
            if time_matrix.matrix[zones[row].drone_id][2] == math.inf:
                time_matrix.update_matrix(self.processing_rate, zones[row].drone_id, 2, t_event)

            # i don't schedule a job arrival, since the job arrived because of offloading (not from job generation)

    # Called each time a job has to exit the processing queue and starts to be processed.
    def job_processing(self, row, t_event, time_matrix, zones):

        p = self.p_queue.pop(0)
        tot_delay = p.get_delay(t_event)
        proc_delay = p.get_processing_delay(t_event)  # set, in the packet, the time in which it has been processed.
        # Useful to gain more insights on time spent in offloading and processing queues
        if p.offloaded:
            off_delay = p.get_offloading_delay()
        else:
            off_delay = None  # we don't want to change the avg_off_delay if the packet is not offloaded
        # save delays in some variable in env
        self.queue -= 1
        self.increase_processed_counter()
        if self.queue == 0:  # start timer for increasing empty queue CE probability
            self.start_timer = t_event
            time_matrix.matrix[row][2] = math.inf
        else:
            time_matrix.update_matrix(self.processing_rate, row, 2, t_event)
        self.processed_pkts += 1

        # updating running delay
        self.running_delay[1] += 1  # counter
        self.running_delay[3] += 1  # counter

        self.running_delay[0] = (((self.running_delay[1] - 1) * self.running_delay[0]) + tot_delay) / \
                                self.running_delay[1]
        self.running_delay[2] = (((self.running_delay[3] - 1) * self.running_delay[2]) + tot_delay) / \
                                self.running_delay[3]

        return tot_delay, proc_delay, off_delay

    # Called each time a job has to be offloaded (exiting the offloading queue). Offload the job to the
    # receiving_drone processing queue
    def job_offloading(self, row, t_event, time_matrix, zones, drones):
        # search the receiving drone
        receiving_drone = search_receiving_drone(drones, row)

        p = self.o_queue.pop(0)
        p.set_offloaded(t_event)  # set, in the packet, the time in which it has been offloaded. Useful to gain more
        # insights on time spent in offloading and processing queues

        self.queue_ol -= 1
        drones[receiving_drone].job_arrival(receiving_drone, t_event, time_matrix, zones, p)
        if self.queue_ol == 0:
            time_matrix.matrix[row][3] = math.inf
        else:
            time_matrix.update_matrix(self.offloading_rate, row, 3, t_event)
        self.offloaded_pkts += 1

    # Called at the end of an episode: retrieves the mean queues lengths
    def get_mean_queue(self):
        mean_q = self.mean_queue_length[0]
        mean_q_ol = self.mean_queue_length[1]

        return mean_q, mean_q_ol

    # increases the counter of the number of jobs arrived in this epoch
    def increase_counter(self):  # called by uav_env when a job arrived by offloading from another drone
        self.job_counter_obs += 1

    # increases the counter of the number of jobs processed in this epoch
    def increase_processed_counter(self):
        self.processed_job_counter_obs += 1

    # Called at the end of an epoch: clear the buffer of the number of jobs arrived / processed and the running delays.
    def clear_buffer(self):
        # don't reset running_delay[0] and [1], since they are episodic
        self.running_delay[2] = 0
        self.running_delay[3] = 0
        self.job_counter_obs = 0
        self.processed_job_counter_obs = 0

    # Could be called by uav_env to change the offloading probability by a certain amount
    def change_offloading_probability(self, amount):
        self.offloading_prob += amount
        if self.offloading_prob > 100:
            self.offloading_prob = 100
        if self.offloading_prob < 0:
            self.offloading_prob = 0

    # Could be called by uav_env to set the offloading probability to a certain amount
    def set_offloading_probability(self, value):
        self.offloading_prob = value * 10
        if self.offloading_prob > 100:
            print("setting off_prob to value > 100%")
            self.offloading_prob = 100
        if self.offloading_prob < 0:
            print("setting off_prob to value < 0")
            self.offloading_prob = 0


# Called when a drone has to offload a packet: search the drone with the least amount of packets in his processing queue
def search_receiving_drone(drones, sending_drone):
    min_queue = K + 1
    destination_drone = None

    for drone in drones:
        if drone.queue < min_queue and drones.index(drone) != sending_drone:
            min_queue = drone.queue
            destination_drone = drone
    return drones.index(destination_drone)


class OtherDrone(Drone):
    def __init__(self, processing_rate, offloading_rate, alg):
        super().__init__(processing_rate, offloading_rate)
        self.alg = alg
        self.drones = None

    def set_drones(self, drones):
        self.drones = drones

    def job_arrival(self, row, t_event, time_matrix, zones, offloaded_packet=None):
        assert self.queue <= K
        assert self.queue_ol <= K_ol
        self.increase_counter()
        self.arrived_pkts += 1
        if offloaded_packet is None:  # packet generated from zone

            # check scheduling algorithm

            if self.alg == "woto":  # local processing
                if self.queue < int(K * 60 / 100) or self.queue_ol >= K_ol:
                    if self.queue < K:
                        self.p_queue.append(Packet(t_event))
                        self.queue += 1

                        if self.queue == 0:  # stop timer for increasing empty queue Computing Element probability
                            self.empty_queue_timer += t_event - self.start_timer

                        if self.queue > self.max_queue_length:
                            self.max_queue_length = self.queue

                        self.mean_queue_length_counter[0] += 1
                        self.mean_queue_length[0] = (((self.mean_queue_length_counter[0] - 1) * self.mean_queue_length[
                            0])
                                                     + self.queue) / self.mean_queue_length_counter[0]
                    else:
                        self.lost_pkts += 1
                    if time_matrix.matrix[zones[row].drone_id][2] == math.inf:  # if the processing queue is empty,
                        # immediately start to process the job
                        time_matrix.update_matrix(self.processing_rate, zones[row].drone_id, 2, t_event)
                else:  # offloading
                    if self.queue_ol < K_ol:
                        self.o_queue.append(Packet(t_event))
                        self.queue_ol += 1
                        if self.queue_ol > self.max_ol_queue_length:
                            self.max_ol_queue_length = self.queue_ol
                        self.mean_queue_length_counter[1] += 1
                        self.mean_queue_length[1] = (((self.mean_queue_length_counter[1] - 1) * self.mean_queue_length[
                            1])
                                                     + self.queue) / self.mean_queue_length_counter[1]
                    else:
                        self.lost_pkts += 1
                    if time_matrix.matrix[zones[row].drone_id][3] == math.inf:  # if the offloading queue is empty,
                        # immediately trigger an offloading event
                        time_matrix.update_matrix(self.offloading_rate, zones[row].drone_id, 3, t_event)

            if self.alg == "fcto":
                min_drone = self.search_min_drone()
                if (self.queue * self.processing_rate) < (self.queue_ol * self.offloading_rate) + \
                        (min_drone.queue * min_drone.processing_rate) or self.queue_ol >= K_ol:  # processing
                    if self.queue < K:
                        self.p_queue.append(Packet(t_event))
                        self.queue += 1

                        if self.queue == 0:  # stop timer for increasing empty queue Computing Element probability
                            self.empty_queue_timer += t_event - self.start_timer

                        if self.queue > self.max_queue_length:
                            self.max_queue_length = self.queue

                        self.mean_queue_length_counter[0] += 1
                        self.mean_queue_length[0] = (((self.mean_queue_length_counter[0] - 1) * self.mean_queue_length[
                            0])
                                                     + self.queue) / self.mean_queue_length_counter[0]
                    else:
                        self.lost_pkts += 1
                    if time_matrix.matrix[zones[row].drone_id][2] == math.inf:  # if the processing queue is empty,
                        # immediately start to process the job
                        time_matrix.update_matrix(self.processing_rate, zones[row].drone_id, 2, t_event)
                else:  # offloading
                    if self.queue_ol < K_ol:
                        self.o_queue.append(Packet(t_event))
                        self.queue_ol += 1
                        if self.queue_ol > self.max_ol_queue_length:
                            self.max_ol_queue_length = self.queue_ol
                        self.mean_queue_length_counter[1] += 1
                        self.mean_queue_length[1] = (((self.mean_queue_length_counter[1] - 1) * self.mean_queue_length[
                            1])
                                                     + self.queue) / self.mean_queue_length_counter[1]
                    else:
                        self.lost_pkts += 1
                    if time_matrix.matrix[zones[row].drone_id][3] == math.inf:  # if the offloading queue is empty,
                        # immediately trigger an offloading event
                        time_matrix.update_matrix(self.offloading_rate, zones[row].drone_id, 3, t_event)

            zones[row].schedule_next_arrival(time_matrix, t_event)
            # schedules the next job arrival (this happens only if the packet was not offloaded )

        else:  # packet needs to be processed locally, and delay needs to be retrieved
            if self.queue < K:
                if self.queue == 0:  # stop timer for increasing empty queue CE probability
                    self.empty_queue_timer += t_event - self.start_timer
                self.p_queue.append(offloaded_packet)
                self.queue += 1
                self.mean_queue_length_counter[0] += 1
                self.mean_queue_length[0] = (((self.mean_queue_length_counter[0] - 1) * self.mean_queue_length[0])
                                             + self.queue) / self.mean_queue_length_counter[0]
            else:
                self.lost_pkts += 1
            if time_matrix.matrix[zones[row].drone_id][2] == math.inf:
                time_matrix.update_matrix(self.processing_rate, zones[row].drone_id, 2, t_event)

            # i don't schedule a job arrival, since the job arrived because of offloading (not from job generation)

    def search_min_drone(self):
        min_queue = K + 1
        destination_drone = None

        for drone in self.drones:
            if drone.queue < min_queue and drone != self:
                min_queue = drone.queue
                destination_drone = drone
        return destination_drone


# Ignore this class. It could be useful in scenarios in which the processing rate has to periodically change
class BatteryDrone(Drone):
    def __init__(self, processing_rate, offloading_rate, start_proc_rate):
        super().__init__(processing_rate, offloading_rate)
        self.max_processing_rate = processing_rate
        self.processing_rate = start_proc_rate
        self.offloading_prob = 0  # not used
        self.inactivity_time = 0
        self.inactivity_start = 0

    def change_processing_rate(self, amount):
        self.processing_rate += self.processing_rate * amount
        if self.processing_rate > self.max_processing_rate:
            self.processing_rate = self.max_processing_rate
        if self.processing_rate <= (0.1 * self.max_processing_rate):
            self.processing_rate = 0.1 * self.max_processing_rate

    def get_job_size(self):
        next_packet_to_process = self.p_queue[0].size
        return next_packet_to_process

    def get_job_delay(self):
        next_packet_to_process = self.p_queue[0].max_delay
        return next_packet_to_process

    def clear_buffer(self):
        super().clear_buffer()
        self.inactivity_time = 0

    def job_arrival(self, row, t_event, time_matrix, zones, offloaded_packet=None):
        assert self.queue <= K
        assert self.queue_ol <= K_ol
        self.increase_counter()
        # CHECK OFF PROB AND DECIDE WHERE TO ENQUEUE THE PCKTs
        decision = random.uniform(0, 1)
        if decision > (self.offloading_prob / 100):  # local processing
            if self.queue < K:
                self.p_queue.append(Packet(t_event))
                if self.queue == 0:  # stop timer for increasing empty queue CE probability
                    self.empty_queue_timer += t_event - self.start_timer
                    self.inactivity_time += t_event - self.inactivity_start
                self.queue += 1
                if self.queue > self.max_queue_length:
                    self.max_queue_length = self.queue
                self.mean_queue_length_counter[0] += 1
                self.mean_queue_length[0] = (((self.mean_queue_length_counter[0] - 1) * self.mean_queue_length[0])
                                             + self.queue) / self.mean_queue_length_counter[0]
            else:
                self.lost_pkts += 1
            self.arrived_pkts += 1

        if time_matrix.matrix[zones[row].drone_id][2] == math.inf:
            was_interrupted = True
        else:
            was_interrupted = False
        # If queue is empty, i will use the processing rate, which still needs to be chosen.
        # Therefore, time_matrix.update_matrix is shifted in evolve_until_processing, which is done
        # after observing the packet size

        zones[row].schedule_next_arrival(time_matrix, t_event)  # schedules the next job arrival
        return was_interrupted

    def job_processing(self, row, t_event, time_matrix, zones):

        p = self.p_queue.pop(0)
        tot_delay = p.get_delay(t_event)  # in the processing queue
        proc_delay = p.get_processing_delay(t_event)
        if p.offloaded:
            off_delay = p.get_offloading_delay()
        else:
            off_delay = None  # we don't want to change the avg_off_delay if the packet is not offloaded
        # save delays in some variable in env
        self.queue -= 1
        self.increase_processed_counter()
        if self.queue == 0:  # start timer for increasing empty queue CE probability
            self.start_timer = t_event
            self.inactivity_start = t_event
            time_matrix.matrix[row][2] = math.inf
        else:
            time_matrix.update_matrix(self.processing_rate, row, 2, t_event, None, job_size=p.size)
        self.processed_pkts += 1

        # updating running delay
        self.running_delay[1] += 1  # counter
        self.running_delay[3] += 1  # counter
        # avg_t1= ((k-1) * avg_t0 + new_t1) / k
        self.running_delay[0] = (((self.running_delay[1] - 1) * self.running_delay[0]) + tot_delay) / \
                                self.running_delay[1]
        self.running_delay[2] = (((self.running_delay[3] - 1) * self.running_delay[2]) + tot_delay) / \
                                self.running_delay[3]

        return tot_delay, proc_delay, off_delay
