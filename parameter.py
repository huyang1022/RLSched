
class Parameter(object):
    def __init__(self):

        self.exp_epochs = 5000                          # number of training epochs
        self.exp_len = 400                             # maximum duration of one experiment
        self.su_epochs = 0                           # supervised training epochs

        self.batch_len = 50                            # maximum duration of one batch
        self.batch_num = 10                            # number of jobs in one batch

        self.res_num = 2                                # number of resources in the cluster
        self.res_slot = 10                            # maximum number of resource slots

        self.mac_num = 3                               # number of machines in the cluster
        self.mac_train_num = self.mac_num               # number of trained machines
        self.mac_max_slot = self.res_slot               # maximum number of resource slots of machine


        self.job_num = 50                               # number of jobs in one batch
        self.job_train_num = 3              # number of trained jobs
        self.job_queue_num = 10000                        # maximum number of jobs in the queue
        self.job_max_len = 20                              # maximum duration of jobs
        self.job_max_slot = self.res_slot * 4 / 5         # maximum number of requested resource
        self.job_interval = 2                            # average inter-arrival time
        self.job_seed =  99                             # random seed for job generating


        self.sched_num = 1                            # number of schedules at one time
        self.sched_flag = False                         # flag of job recycle
        self.ecs_sched_num = 5                           # number of schedules at one time (ecs)
        self.ecs_process_num = 100                       # number of jobs that can be processed in ecs scheduler
        self.agent = "rl"

        self.alg_num = 3                                # number of candidate algorithms
        self.a_learn_rate = 0.0001                         # actor learning rate
        self.c_learn_rate = 0.0001                         # critic learning rate
        self.discount_rate = 0.9                         # discount rate
        self.learn_step = 100                            # steps of update
        self.eps = 1e-10                                 # eps
        self.entropy_rate = 0.001
        self.save_step = 10                               # parameters save step

        # usage: job_max_len  / interval / mac_num / 10