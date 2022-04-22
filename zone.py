LOW_ACTIVITY = 0
HIGH_ACTIVITY = 1


class Zone:
    def __init__(self, id_n, lmbda_h, lmbda_l, drone_id):
        self.id = id_n
        # self.zone_size = area_size / number_of_drones  # size of the zone monitored by the UAV
        self.drone_id = drone_id
        self.state = LOW_ACTIVITY
        # self.drones_present = [z, 0]
        self.lmbda_h = lmbda_h
        self.lmbda_l = lmbda_l
        self.emission_rate = lmbda_l if self.state == 0 else lmbda_h
        self.inter_arrival_time_zone = 0

    def change_zone_state(self, t_event, time_matrix):
        if self.state == LOW_ACTIVITY:
            self.state = HIGH_ACTIVITY
            self.emission_rate = self.lmbda_h
        else:
            self.state = LOW_ACTIVITY
            self.emission_rate = self.lmbda_l

        time_matrix.update_matrix(0, self.id, 0, t_event, self.state)  # schedule next state change

    def schedule_next_arrival(self, time_matrix, t_event):
        time_matrix.update_matrix(0, self.id, 1, t_event, self.state)  # schedule next job arrival

    def increase_lmbda_high(self):
        self.lmbda_h += 0.1
