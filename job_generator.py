from element import Job
from parameter import Parameter
import numpy as np

class JobGenerator(object):
    def __init__(self, pa):
        # type: (Parameter) -> object
        self.job_sequence = []
        self.job_dag_id = []
        self.dag_matrix = []
        self.total_len = 0.0

        self.short_rate = 0.5
        #
        # self.long_upper = pa.job_max_len
        # self.long_lower = pa.job_max_len  / 2 + 1
        # self.short_upper = pa.job_max_len / 2
        # self.short_lower = 1
        self.long_upper = pa.job_max_len
        self.long_lower = 1
        self.short_upper = pa.job_max_len
        self.short_lower = 1


        # self.dominant_upper = pa.job_max_slot
        # self.dominant_lower = pa.job_max_slot / 2 + 1
        # self.other_upper = pa.job_max_slot / 2
        # self.other_lower = 1

        self.dominant_upper = pa.job_max_slot
        self.dominant_lower = 1
        self.other_upper = pa.job_max_slot
        self.other_lower = 1


        if pa.test_flag:
            self.batch_num = pa.batch_num * 2
        else:
            self.batch_num = pa.batch_num

        np.random.seed(pa.job_seed)

        for i in xrange(pa.dag_num):
            self.dag_matrix.append([])
            dag_name = pa.dag_dict[i][0]
            dag_file = open("./dag/%s" % dag_name, "r")
            for line in dag_file.readlines():
                self.dag_matrix[i].append([int(x) for x in line.split()])



        if pa.dag_id != None:
            for k in xrange(self.batch_num):
                if pa.dag_id == 9:
                    self.job_dag_id.append((k % 3) + 6)
                else:
                    self.job_dag_id.append(pa.dag_id)

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


            for k in xrange(self.batch_num):
                for i in xrange(pa.job_num):
                    self.job_sequence[k][i].depth, self.job_sequence[k][i].c_next, self.job_sequence[k][i].c_len  = self.dfs(k, i)

            for k in xrange(self.batch_num):
                for i in xrange(pa.job_num):
                    self.job_sequence[k][i].c_state = np.zeros([pa.dag_max_depth, pa.job_max_len])
                    job = self.job_sequence[k][i]
                    self.job_sequence[k][i].c_state[job.depth, :job.duration] = 1
                    j = job.c_next
                    while j != -1:
                        job = self.job_sequence[k][j]
                        self.job_sequence[k][i].c_state[job.depth, :job.duration] = 1
                        j = job.c_next


                # self.job_sequence[k].sort(key=lambda x: (- x.depth, - x.c_len, x.id))






    def dfs(self, b_id, j_id):
        ret_depth = -1
        ret_next = -1
        ret_len = 0
        dag_id = self.job_dag_id[b_id]
        for i in xrange(len(self.dag_matrix[dag_id][j_id])):
            if self.dag_matrix[dag_id][j_id][i]:
                n_depth, n_next, n_len = self.dfs(b_id, i)
                if n_depth > ret_depth:
                    ret_depth = n_depth
                if n_len > ret_len:
                    ret_len = n_len
                    ret_next = i
        return ret_depth + 1, ret_next, ret_len + self.job_sequence[b_id][j_id].duration
