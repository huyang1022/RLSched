from element import Action
import sys


def run(env, act_id):
    MOD = env.pa.mac_train_num
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

def get_id(env, alg_id):
    MOD = env.pa.mac_train_num
    if env.job_count == 0: return env.pa.mac_train_num * env.pa.job_train_num
    job_n = min(env.pa.job_train_num, env.job_count)
    ret_act = env.pa.mac_train_num * env.pa.job_train_num


    if alg_id % env.pa.alg_num == 0:    #pack
        max_score = 0
        for i in xrange(job_n):
            for j in xrange(env.mac_count):
                act = Action(env.jobs[i].id, env.macs[j].id)
                if env.check_act(act):
                    score = 0
                    for k in xrange(env.pa.res_num):
                        res_avail = (env.macs[j].state[k] == 0)
                        score += res_avail.sum() * env.jobs[i].res_vec[k]
                    if score > max_score:
                        max_score = score
                        ret_act = MOD * i + j
    elif alg_id % env.pa.alg_num == 1:  #sjf
        min_duration = sys.maxint
        for i in xrange(job_n):
            for j in xrange(env.mac_count):
                act = Action(env.jobs[i].id, env.macs[j].id)
                if env.check_act(act) and env.jobs[i].duration < min_duration:
                    ret_act = MOD * i + j
                    min_duration = env.jobs[i].duration
    elif alg_id % env.pa.alg_num == 2:     #cp
        max_len = 0
        for i in xrange(job_n):
            for j in xrange(env.mac_count):
                act = Action(env.jobs[i].id, env.macs[j].id)
                if env.check_act(act) and env.jobs[i].c_len > max_len:
                    max_len = env.jobs[i].c_len
                    ret_act = MOD * i + j
                # elif env.jobs[i].c_len > max_len:
                #     max_len = env.jobs[i].c_len
                #     ret_act = MOD * i + j
    return ret_act


