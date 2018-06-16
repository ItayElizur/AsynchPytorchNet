import random
import time


class Partition():

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


# Partition and distribute dataset to workers
class DataPartitioner():

    def __init__(self, data, num_workers, seed=time.time()):
        self.data = data
        self.partitions = []
        num_data = len(data)
        data_indexes = [x for x in range(0, num_data)]
        # Randomize which sets each worker recieves
        rng = random.Random()
        rng.seed(seed)
        rng.shuffle(data_indexes)
        part_len = int(num_data / num_workers)
        # Assign datasets
        for worker in range(num_workers):
            self.partitions.append(data_indexes[0:part_len])
            data_indexes = data_indexes[part_len:]

    def assign(self, worker_num):
        return Partition(self.data, self.partitions[worker_num])
