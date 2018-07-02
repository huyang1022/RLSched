import numpy as np

from parameter import Parameter
from element import Machine, Job, Action

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


        self.time_file = open("log/%s" % (pa.agent), "w")   #file to record the logs
        self.job_file = open("data/%s" % (pa.agent), "w")  # file to record the logs

    def reset(self):
        self.current_time = -1
        self.macs = []
        self.mac_count = 0
        self.jobs = []
        self.job_count = 0
        self.running_jobs = []
        self.finished_jobs = []

    def add_machine(self, mac):
        # type: (Machine) -> None
        self.macs.append(mac)
        self.mac_count += 1
        assert self.mac_count <= self.pa.mac_num


    def add_job(self, job):
        # type: (Job) -> None
        self.jobs.append(job)
        self.job_count += 1
        assert self.job_count <= self.pa.job_queue_num

    def pop_job(self, job_id):
        job_index = [x.id for x in self.jobs].index(job_id)
        self.jobs.pop(job_index)
        self.job_count -= 1

    def check_act(self, act): #act = [job_x, mac_y]  allocate job x to machine y
        # type: (Action) -> None
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
        ret_list = []
        for mac in self.macs:
            ret_list.append(mac.state)
        for job in self.jobs:
            ret_list.append(job.state)
        for i in xrange(self.pa.job_queue_num - self.job_count):
            ret_list.append(np.zeros([self.pa.res_num, self.pa.res_slot]))
        return np.concatenate(ret_list)


    def step(self): #act = [job_x, mac_y]  allocate job x to machine y
        # type: (Environment) -> None
        self.current_time += 1
        for mac in self.macs:
            mac.step()

        for job in self.running_jobs:
            job.step()
            # if job.status == "Finished":
            #     print "ID: ",               job.id
            #     print "Submission Time: ",  job.submission_time
            #     print "Starting Time",      job.starting_time
            #     print "Execution time: ",   job.execution_time
            #     print "======================================="
        self.finished_jobs.extend([job for job in self.running_jobs if job.status == "Finished"])
        self.running_jobs = [job for job in self.running_jobs if job.status != "Finished"]

    def status(self):
        # type: (Environment) -> str
        if self.running_jobs:
            return "Running"
        if self.jobs:
            return "Pending"
        return "Idle"

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
