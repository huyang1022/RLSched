
class Parameter(object):
    def __init__(self):

        self.exp_epochs = 10000                          # number of training epochs
        self.exp_len = 1000                             # maximum duration of one experiment

        self.batch_len = 200                            # maximum duration of one batch
        self.batch_num = 1                            # number of jobs in one batch

        self.res_num = 2                                # number of resources in the cluster
        self.res_slot = 10                            # maximum number of resource slots

        self.mac_num = 10                               # number of machines in the cluster
        self.mac_max_slot = self.res_slot               # maximum number of resource slots of machine


        self.job_num = 100                               # number of jobs in one batch
        self.job_queue_num = 100                        # maximum number of jobs in the queue
        self.job_max_len = self.batch_len * 4 / 5         # maximum duration of jobs
        self.job_max_slot = self.res_slot * 4 / 5         # maximum number of requested resource
        self.job_interval = 2                            # average inter-arrival time
        self.job_seed = 42                              # random seed for job generating


        self.sched_num = 1                            # number of schedules at one time
        self.sched_flag = False                         # flag of job recycle
        self.ecs_sched_num = 5                           # number of schedules at one time (ecs)
        self.ecs_process_num = 100                       # number of jobs that can be processed in ecs scheduler
        self.agent = "none"

        self.alg_num = 4                                # number of candidate algorithms
        self.learn_rate = 0.0001                         # learning rate
        self.discount_rate = 1.0                         # discount rate
        self.learn_step = 100                            # steps of update
        self.eps = 1e-10                                 # eps
        self.entropy_rate = 0.001
        self.save_step = 10                               # parameters save step

        # usage: job_max_len  / interval / mac_num / 10