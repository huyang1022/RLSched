import numpy as np
from parameter import Parameter
from element import Machine, Job, Action
import act_generator

class Environment(object):
    def __init__(self, pa):
        # type: (Parameter) -> object
        self.pa = pa
        self.current_time = -1

        self.macs = []          # set of machines in the cluster
        self.mac_count = 0      # number of machines
        self.jobs = []          # set of jobs in the queue
        self.job_count = 0      # number of jobs
        self.running_jobs = []  # set of running jobs
        self.finished_jobs = [] # set of finished jobs
        self.finished_ids = []  # ID of finished jobs
        self.batch_id = 0       # ID of batch
        self.job_gen = None     # job generator
        self.job_gen_idx = 0    # index of job_gen
        self.mac_gen = None     # mac generator
        self.job_set = set()   # flag of enqueue
        # self.time_file = open("./log/env_time_%s" % (pa.agent), "w")   #file to record the logs
        # self.job_file = open("./log/env_job_%s" % (pa.agent), "w")  # file to record the logs


    def reset(self):
        self.current_time = -1
        self.macs = []
        self.mac_count = 0
        self.jobs = []
        self.job_count = 0
        self.running_jobs = []
        self.finished_jobs = []
        self.finished_ids = []
        self.batch_id = 0
        self.job_gen_idx = 0
        self.job_set = set()

    def add_machine(self, mac):
        # type: (Machine) -> None
        self.macs.append(mac)
        self.mac_count += 1
        assert self.mac_count <= self.pa.mac_num

    def check_machine(self, mac_id):
        for i in self.macs:
            if i.id == mac_id: return True
        return False


    def add_cluster(self):
        for i in self.mac_gen.mac_sequence:
            self.add_machine(i)

    def add_job(self, job):
        # type: (Job) -> None
        self.jobs.append(job)
        self.job_count += 1
        assert self.job_count <= self.pa.job_queue_num

    def pop_job(self, job_id):
        job_index = [x.id for x in self.jobs].index(job_id)
        self.jobs.pop(job_index)
        self.job_count -= 1


    def check_job(self, job_id):
        for i in self.jobs:
            if i.id == job_id: return True
        return False

    def check_parent(self, job_id):
        for i in xrange(self.pa.job_num):
            if self.job_gen.job_matrix[i][job_id]:
                if i not in self.finished_ids:
                    return False
        return True

    def check_act(self, act): #act = [job_x, mac_y]  allocate job x to machine y
        # type: (Action) -> int
        if not self.check_machine(act.mac_id): return False
        if not self.check_job(act.job_id): return False
        job_index = [x.id for x in self.jobs].index(act.job_id)
        mac_index = [x.id for x in self.macs].index(act.mac_id)
        for i in xrange(self.pa.res_num):
            res_avail = (self.macs[mac_index].state[i] == 0)
            if res_avail.sum() < self.jobs[job_index].res_vec[i]:
                return False
        return True

    def take_act(self, act):
        # type: (Action) -> None
        job_index = [x.id for x in self.jobs].index(act.job_id)
        mac_index = [x.id for x in self.macs].index(act.mac_id)
        for i in xrange(self.pa.res_num):
            res_avail = (self.macs[mac_index].state[i] == 0)
            assert res_avail.sum() >= self.jobs[job_index].res_vec[i]
            self.macs[mac_index].state[i, res_avail] = self.jobs[job_index].state[i, :res_avail.sum()]

        self.running_jobs.append(self.jobs[job_index])
        self.running_jobs[-1].start(self.current_time)

    def obs(self):
        mac_n = min(self.pa.mac_train_num, self.mac_count)
        job_n = min(self.pa.job_train_num, self.job_count)
        m_obs = np.zeros([self.pa.mac_train_num, self.pa.res_num, self.pa.mac_max_slot, self.pa.job_max_len])
        j_obs = np.zeros([self.pa.job_train_num, self.pa.res_num, self.pa.job_max_slot, self.pa.job_max_len])
        d_obs = np.zeros([self.pa.job_train_num, self.pa.dag_max_depth * self.pa.job_max_len])

        for i in xrange(mac_n):
            for j in xrange(self.pa.res_num):
                for k in xrange(self.pa.mac_max_slot):
                    m_obs[i][j][k][:int(self.macs[i].state[j][k])] = 1

        for i in xrange(job_n):
            for j in xrange(self.pa.res_num):
                for k in xrange(self.pa.job_max_slot):
                    j_obs[i][j][k][:int(self.jobs[i].state[j][k])] = 1

            d_obs[i][:self.jobs[i].depth] = 1
            d_obs[i][self.pa.dag_max_depth:self.pa.dag_max_depth + self.jobs[i].c_len] = 1

        return np.concatenate((m_obs.flatten(), j_obs.flatten(), d_obs.flatten()))

    def reward(self):
        # return -self.job_count * 1.0 / self.pa.job_num
        # return self.current_time * -1.0 / self.pa.batch_len
        return -1.0
    def step(self): #act = [job_x, mac_y]  allocate job x to machine y
        # type: (Environment) -> None
        self.current_time += 1
        for mac in self.macs:
            mac.step()

        for job in self.running_jobs:
            job.step()

        self.finished_jobs.extend([job for job in self.running_jobs if job.status == "Finished"])
        self.finished_ids.extend([job.id for job in self.running_jobs if job.status == "Finished"])
        self.running_jobs = [job for job in self.running_jobs if job.status != "Finished"]

    def status(self):
        # type: (Environment) -> str
        if self.running_jobs:
            return "Running"
        if self.jobs:
            return "Pending"
        return "Idle"

    def step_act(self, act_id):
        act = act_generator.run(self, act_id)

        if act is not None:
            self.take_act(act)
            self.pop_job(act.job_id)
            ret_state = self.obs()
            ret_reward = 0.0
            ret_done = False
            ret_info = 0
            ret_flag = 1
        else:
            self.step()
            for job in self.job_gen.job_sequence[self.batch_id]:
                if job.id not in self.job_set:
                    if self.check_parent(job.id):
                        self.job_set.add(job.id)
                        self.add_job(job)

            # job = self.job_gen.job_sequence[self.batch_id][self.job_gen_idx]
            # while (job is not None and
            #        job.submission_time <= self.current_time
            #     ):
            #     if (job.submission_time == self.current_time):  # add job to environment
            #         self.add_job(job)
            #     self.job_gen_idx += 1
            #     job = self.job_gen.job_sequence[self.batch_id][self.job_gen_idx]
            ret_state = self.obs()
            ret_flag = 0
            if len(self.finished_jobs) == self.pa.job_num:
                ret_reward = 0.0
                ret_info = 0
                ret_done = True
            else:
                ret_reward = self.reward()
                ret_info = 1
                ret_done = False

        return ret_state, ret_reward, ret_done, ret_info, ret_flag

    def get_usage(self, res_id):
        res_used = 0
        res_total = 0
        for i in self.macs:
            res_total += (i.state[res_id] >= 0).sum()
            res_used += (i.state[res_id] > 0).sum()
        return res_used * 1.0 / res_total

    def time_log(self):
        running_num = len(self.running_jobs)
        finished_num = len(self.finished_jobs)
        res_usage = []
        for i in xrange(self.pa.res_num):
            res_usage.append(self.get_usage(i))


        print "== Agent:%s, Time:%d, Pend:%d, Run:%d, Finish:%d, Cpu:%f, Mem:%f =="  %(self.pa.agent, self.current_time, self.job_count, running_num, finished_num, res_usage[0], res_usage[1])
        self.time_file.write( "Agent %s Time %d Pend %d Run %d Finish %d Cpu %f Mem %f \n" %(self.pa.agent, self.current_time, self.job_count, running_num, finished_num, res_usage[0], res_usage[1]))
        # print "Machine: =========================================================="
        # for i in xrange(self.mac_count):
        #     self.macs[i].show()
        # print "Job: ==============================================================\n"
        # for i in xrange(self.job_count):
        #     self.jobs[i].show()


    def job_log(self):
        self.finished_jobs.sort(key=lambda x: x.id)
        for i in self.finished_jobs:
            self.job_file.write("%d %d \n" % (i.id, i.starting_time - i.submission_time + i.execution_time))

    def finish(self):
        self.time_file.close()
        self.job_file.close()
