import random


class Packet:
    def __init__(self, generation_time):
        self.generation_time = generation_time  # used to retrieve the final delay
        self.offloaded = False  # used to understand if the packet has been offloaded or not
        self.offloading_delay = 0  # used, when the packet is offloaded, to store the offloading delay

        self.size = random.uniform(0.7, 1.3)  # used in BatteryTimeMatrix to change the size of the packet. Ignore it
        self.max_delay = 10  # not used and not implemented. Could be used in scenarios in which the job has a maximum
        # delay requirement.

    # Called when a packet get processed: returns the experienced delay
    def get_delay(self, processing_time):
        return processing_time - self.generation_time

    # Called when a packet exits the offloading queue: returns the offloading delay
    def get_offloading_delay(self):
        if self.offloaded:
            return self.offloading_delay
        else:
            print("Trying to get offloading delay of a non-offloaded packet")

    # Called when a packet exits the processing queue: returns the processing delay (maybe it doesn't consider the
    # time required to be really processed, but only the time spent in the processing queue)
    def get_processing_delay(self, processing_time):
        if self.offloaded:
            delay = processing_time - self.offloading_delay - self.generation_time
        else:
            delay = processing_time - self.generation_time

        return delay

    # Called then the packet is offloaded. Mandatory to get the real processing delay in get_processing_delay
    def set_offloaded(self, offloading_time):
        self.offloaded = True
        self.offloading_delay = offloading_time - self.generation_time
