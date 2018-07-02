
class Parameter(object):
    def __init__(self):

        self.exp_len = 1000                                # maximum duration of one experiment
        self.exp_num = 500                                 # num of experiments


        self.res_num = 2                                # number of resources in the cluster
        self.res_slot = 10                            # maximum number of resource slots

        self.mac_num = 10                               # number of machines in the cluster
        self.mac_max_slot = self.res_slot               # maximum number of resource slots of machine


        self.job_queue_num = 1000                        # maximum number of jobs in the queue
        self.job_process_num = 100                        # maximum number of jobs that can be processed at one time (qps)
        self.job_max_len = self.exp_len * 2 / 3                # maximum duration of jobs
        self.job_max_slot = self.res_slot               # maximum number of requested resource
        self.job_interval = 4                           # average inter-arrival time
        self.job_seed = 66                              # random seed for job generating


        self.sched_num = 1                            # number of schedules at one time
        self.sched_flag = False                         # flag of job recycle
        self.ecs_num = 1                                # maximum number of jobs that can be processed in ecs scheduler
        self.agent = "None"

        self.alg_num = 4                                # number of candidate algorithms
        self.learn_rate = 0.001                         # learning rate
        self.discount_rate = 0.99                               # discount rate

        # usage: job_max_len  / interval / mac_num / 10