
class Parameter(object):
    def __init__(self):

        self.exp_epochs = 50000                          # number of training epochs
        self.exp_len = 500                             # maximum duration of one experiment
        self.su_epochs = 100                           # supervised training epochs

        self.batch_len = 50                            # maximum duration of one batch
        self.batch_num = 5                            # number of jobs in one batch

        self.res_num = 2                                # number of resources in the cluster
        self.res_slot = 10                            # maximum number of resource slots

        self.mac_num = 2                               # number of machines in the cluster
        self.mac_train_num = self.mac_num               # number of trained machines
        self.mac_max_slot = self.res_slot               # maximum number of resource slots of machine


        self.job_num = 50                               # number of jobs in one batch
        self.job_train_num = 5                          # number of trained jobs
        self.job_queue_num = 10000                        # maximum number of jobs in the queue
        self.job_max_len = 20                              # maximum duration of jobs
        self.job_max_slot = self.res_slot * 4 / 5         # maximum number of requested resource
        self.job_interval = None                            # average inter-arrival time
        self.job_seed = 7                             # random seed for job generating

        self.dag_num = 6                                # number of dag
        self.dag_id = 0                                # id of dag
        self.dag_dict = {                               # name of dag file and number of jobs
            0: ["Epigenomics_50", 50],
            1: ["CyberShake_50", 50],
            2: ["Montage_50", 50],
            3: ["Epigenomics_100", 100],
            4: ["CyberShake_100", 100],
            5: ["Montage_100", 100],
            6: ["test", 5]
        }
        self.dag_max_depth = 10                         # max depth of a dag

        if self.dag_id != None:
            self.job_num = self.dag_dict[self.dag_id][1]

        self.sched_num = 1                            # number of schedules at one time
        self.sched_flag = False                         # flag of job recycle
        self.ecs_sched_num = 5                           # number of schedules at one time (ecs)
        self.ecs_process_num = 100                       # number of jobs that can be processed in ecs scheduler
        self.agent = "rl"

        self.alg_num = 3                                # number of candidate algorithms
        self.a_learn_rate = 0.0001                         # actor learning rate
        self.c_learn_rate = 0.001                         # critic learning rate
        self.discount_rate = 0.99                         # discount rate
        self.learn_step = 30                            # steps of update
        self.eps = 1e-10                                 # eps
        self.entropy_rate = 0.03
        self.save_step = 10                               # parameters save step
        self.test_flag = True                             # flag of test

        # usage: job_max_len  / interval / mac_num / 10