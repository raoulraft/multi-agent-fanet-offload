import math
import numpy as np
from utils import esp_rand, esp_rand_zone


class TimeMatrix:
    def __init__(self, n_max, prob_trans, lambdas):
        dim_matrix = n_max
        self.matrix = np.zeros((dim_matrix, 4))
        self.prob_trans = prob_trans
        self.lambdas = lambdas
        for i in range(dim_matrix):
            # low priority events
            self.matrix[i, 0] = esp_rand_zone(self.prob_trans[0])
            # print(time_matrix[i-1,0])
            self.matrix[i, 1] = esp_rand(self.lambdas[0])  # LOW PRIORITY ARRIVAL EVENT
            self.matrix[i, 2] = math.inf  # since queues are empty, there is no scheduled processing event
            self.matrix[i, 3] = math.inf  # since queues are empty, there is no scheduled offloading event

    def search_next_event(self):
        t_event = np.amin(np.amin(self.matrix))  # get the min element among the ones in the time matrix
        # return index and column of the event
        index = np.where(self.matrix == t_event)
        [row, column] = [index[0][0], index[1][0]]  # get row and column from index
        return row, column, t_event

    def update_matrix(self, mu_drone, row, column, t_event, zone_state=None):
        """
        :param mu_drone: based on the column (event), it can either be mu_p high/low or mu_ol (or zero)
        :param row: drone/zone being affected by the event
        :param column: type of event
        :param t_event: time of event
        :param zone_state: zones
        :return:
        """
        job_size = 0

        zone_number = row  # get the number of the zone

        if column == 0:  # EVENT = CHANGE ACTIVITY STATE OF THE ZONE
            if zone_state == 0:  # if zone.state == 0
                average_residence_time = 1 / self.prob_trans[0]  # average residence time in LOW ACTIVITY state
            else:
                average_residence_time = 1 / self.prob_trans[1]  # average residence time in HIGH ACTIVITY state

            lambda_zone = 1 / average_residence_time  # zone's lambda
            self.matrix[zone_number, 0] = t_event + esp_rand_zone(
                lambda_zone)  # GET TIME OF THE NEXT CHANGE ACTIVITY STATE OF THE ZONE

        elif column == 1:  # EVENT = JOB ARRIVAL
            if zone_state == 0:  # if zone in LOW ACTIVITY state
                self.matrix[zone_number, 1] = t_event + esp_rand(self.lambdas[0])

            else:  # if zone in HIGH ACTIVITY state
                self.matrix[zone_number, 1] = t_event + esp_rand(self.lambdas[1])

        elif column == 2:  # EVENT = JOB PROCESSING

            self.matrix[zone_number, 2] = t_event + esp_rand(mu_drone)

        elif column == 3:  # EVENT = JOB OFFLOADING
            job_size = esp_rand(mu_drone)
            self.matrix[zone_number, 3] = t_event + job_size


class BatteryTimeMatrix(TimeMatrix):
    def __init__(self, n_max, prob_trans, lambdas):
        super().__init__(n_max, prob_trans, lambdas)

    def update_matrix(self, mu_drone, row, column, t_event, zone_state=None, job_size=None):

        zone_number = row  # get the number of the zone

        if column == 0:  # EVENT = CHANGE ACTIVITY STATE OF THE ZONE
            if zone_state == 0:  # if zone.state == 0
                average_residence_time = 1 / self.prob_trans[0]  # average residence time in LOW ACTIVITY state
            else:
                average_residence_time = 1 / self.prob_trans[1]  # average residence time in HIGH ACTIVITY state

            lambda_zone = 1 / average_residence_time  # zone's lambda
            self.matrix[zone_number, 0] = t_event + esp_rand_zone(
                lambda_zone)  # GET TIME OF THE NEXT CHANGE ACTIVITY STATE OF THE ZONE

        elif column == 1:  # EVENT = JOB ARRIVAL
            if zone_state == 0:  # if zone in LOW ACTIVITY state
                self.matrix[zone_number, 1] = t_event + esp_rand(self.lambdas[0])

            else:  # if zone in HIGH ACTIVITY state
                self.matrix[zone_number, 1] = t_event + esp_rand(self.lambdas[1])

        elif column == 2:  # EVENT = JOB PROCESSING

            self.matrix[zone_number, 2] = t_event + esp_rand(mu_drone/job_size)

        elif column == 3:  # EVENT = JOB OFFLOADING
            job_size = esp_rand(mu_drone)
            self.matrix[zone_number, 3] = t_event + job_size
