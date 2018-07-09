from element import Action
import sys


def run(env, act_id):
    MOD = env.pa.mac_train_num
    # return sjf1(env)
    if act_id  == env.pa.mac_train_num * env.pa.job_train_num: return  None
    mac_idx = act_id % MOD
    job_idx = act_id // MOD
    if mac_idx >= env.mac_count: return  None
    if job_idx >= env.job_count: return  None
    act = Action(env.jobs[job_idx].id, env.macs[mac_idx].id)
    if env.check_act(act):
        return act
    else:
        return None

def get_id(env):
    MOD = env.pa.mac_train_num
    if env.job_count == 0: return env.pa.mac_train_num * env.pa.job_train_num
    job_n = min(env.pa.job_train_num, env.job_count)
    min_duration = sys.maxint
    # #
    job_idx = -1
    for i in xrange(job_n):
        if env.jobs[i].duration < min_duration:
            min_duration = env.jobs[i].duration
            job_idx = i

    for j in xrange(env.mac_count):
        act = Action(env.jobs[job_idx].id, env.macs[j].id)
        if env.check_act(act):
            return MOD * job_idx + j
    return env.pa.mac_train_num * env.pa.job_train_num


    ret_act = env.pa.mac_train_num * env.pa.job_train_num
    for i in xrange(job_n):
        for j in xrange(env.mac_count):
            act = Action(env.jobs[i].id, env.macs[j].id)
            if env.check_act(act) and env.jobs[i].duration < min_duration:
                ret_act = MOD * i + j
                min_duration = env.jobs[i].duration
    return ret_act




