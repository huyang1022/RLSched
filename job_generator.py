from element import Job
from parameter import Parameter
import numpy as np

class JobGenerator(object):
    def __init__(self, pa):
        # type: (Parameter) -> object
        self.job_sequence = []
        self.rate = 1.0 / pa.job_interval

        self.short_rate = 0.8

        self.long_upper = pa.job_max_len * 2 / 3
        self.long_lower = pa.job_max_len  / 3
        self.short_upper = pa.job_max_len / 4
        self.short_lower = 1

        self.dominant_upper = pa.job_max_slot * 2 / 3
        self.dominant_lower = pa.job_max_slot / 3
        self.other_upper = pa.job_max_slot / 4
        self.other_lower = 1

        np.random.seed(pa.job_seed)
        for i in xrange(pa.exp_len):
            if np.random.rand() <= self.rate:                    # generate a new job
                if np.random.rand() <= self.short_rate:          # generate a short job
                    duration = np.random.randint(self.short_lower, self.short_upper + 1)
                else:
                    duration = np.random.randint(self.long_lower, self.long_upper + 1)


                donimant_res = np.random.randint(0, pa.res_num)
                res_vec = []
                for j in xrange(pa.res_num):
                    if j == donimant_res:
                        res_vec.append(np.random.randint(self.dominant_lower, self.dominant_upper + 1))
                    else:
                        res_vec.append(np.random.randint(self.other_lower, self.other_upper + 1))

                self.job_sequence.append(Job(i, duration, pa.res_num, pa.res_slot, res_vec))


        # in_file = open("Input", "r")
        # while True:
        #     line = in_file.readline()
        #     line = line.split()
        #     if not line: break
        #     self.job_sequence.append(Job(int(line[1]), int(line[2]), pa.res_num, pa.res_slot, [int(line[3]), int(line[4])]))



        self.job_sequence.append(None)