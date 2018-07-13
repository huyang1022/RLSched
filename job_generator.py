from element import Job
from parameter import Parameter
import numpy as np

class JobGenerator(object):
    def __init__(self, pa):
        # type: (Parameter) -> object
        self.job_sequence = []
        self.job_matrix = []
        self.total_len = 0.0

        self.short_rate = 0.8

        self.long_upper = pa.job_max_len
        self.long_lower = pa.job_max_len  / 2
        self.short_upper = pa.job_max_len / 4
        self.short_lower = 1

        self.dominant_upper = pa.job_max_slot
        self.dominant_lower = pa.job_max_slot / 2
        self.other_upper = pa.job_max_slot / 4
        self.other_lower = 1

        np.random.seed(pa.job_seed)
        if pa.dag_id != None:
            dag_name = pa.dag_dict[pa.dag_id][0]
            dag_file = open("./dag/%s" % dag_name, "r")
            for line in dag_file.readlines():
                self.job_matrix.append([int(x) for x in line.split()])

            pa.job_num = pa.dag_dict[pa.dag_id][1]
            for k in xrange(pa.batch_num):
                self.job_sequence.append([])
                for i in xrange(pa.job_num):
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

                    self.total_len += duration
                    self.job_sequence[k].append(Job(0, duration, pa.res_num, pa.res_slot, res_vec, i))


            for k in xrange(pa.batch_num):
                for i in xrange(pa.job_num):
                    self.job_sequence[k][i].depth, self.job_sequence[k][i].c_next, self.job_sequence[k][i].c_len  = self.dfs(k, i)

            for k in xrange(pa.batch_num):
                for i in xrange(pa.job_num):
                    self.job_sequence[k][i].c_state = np.zeros([pa.dag_max_depth, pa.job_max_len])
                    job = self.job_sequence[k][i]
                    self.job_sequence[k][i].c_state[job.depth, :job.duration] = 1
                    j = job.c_next
                    while j != -1:
                        job = self.job_sequence[k][j]
                        self.job_sequence[k][i].c_state[job.depth, :job.duration] = 1
                        j = job.c_next


                self.job_sequence[k].sort(key=lambda x: (- x.depth, - x.c_len, x.id))

        elif pa.job_num != None:
            for k in xrange(pa.batch_num):
                self.job_sequence.append([])
                for i in xrange(pa.job_num):
                    submission_time = np.random.randint(0, pa.batch_len)
                    if np.random.rand() <= self.short_rate:  # generate a short job
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

                    self.total_len += duration
                    self.job_sequence[k].append(Job(0, duration, pa.res_num, pa.res_slot, res_vec, i))

                self.job_sequence[k].sort(key=lambda x: x.submission_time)
                self.job_sequence[k].append(None)

        # self.rate = 1.0 / pa.job_interval
        # for i in xrange(pa.exp_len):
        #     if np.random.rand() <= self.rate:                    # generate a new job
        #         if np.random.rand() <= self.short_rate:          # generate a short job
        #             duration = np.random.randint(self.short_lower, self.short_upper + 1)
        #         else:
        #             duration = np.random.randint(self.long_lower, self.long_upper + 1)
        #
        #
        #         donimant_res = np.random.randint(0, pa.res_num)
        #         res_vec = []
        #         for j in xrange(pa.res_num):
        #             if j == donimant_res:
        #                 res_vec.append(np.random.randint(self.dominant_lower, self.dominant_upper + 1))
        #             else:
        #                 res_vec.append(np.random.randint(self.other_lower, self.other_upper + 1))
        #
        #         self.job_sequence.append(Job(i, duration, pa.res_num, pa.res_slot, res_vec))
        # self.job_sequence.append(None)

        # in_file = open("Input", "r")
        # while True:
        #     line = in_file.readline()
        #     line = line.split()
        #     if not line: break
        #     self.job_sequence.append(Job(int(line[1]), int(line[2]), pa.res_num, pa.res_slot, [int(line[3]), int(line[4])]))
        # self.job_sequence.append(None)


    def dfs(self, b_id, j_id):
        ret_depth = -1
        ret_next = -1
        ret_len = 0

        for i in xrange(len(self.job_matrix[j_id])):
            if self.job_matrix[j_id][i]:
                n_depth, n_next, n_len = self.dfs(b_id, i)
                if n_depth > ret_depth:
                    ret_depth = n_depth
                if n_len > ret_len:
                    ret_len = n_len
                    ret_next = i
        return ret_depth + 1, ret_next, ret_len + self.job_sequence[b_id][j_id].duration
