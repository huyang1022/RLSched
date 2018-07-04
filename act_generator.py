from element import Action
import sys

# def fifo(env):
#     for i in xrange(env.job_count):
#         for j in xrange(env.mac_count):
#             act = Action(env.jobs[i].id, env.macs[j].id)
#             if env.check_act(act):
#                 return act
#     return None

#
# def sjf(env):
#     ret_act = None
#     min_duration = sys.maxint
#     for i in xrange(env.job_count):
#         for j in xrange(env.mac_count):
#             act = Action(env.jobs[i].id, env.macs[j].id)
#             if env.check_act(act) and env.jobs[i].duration < min_duration:
#                 ret_act = act
#                 min_duration = env.jobs[i].duration
#     return ret_act
# #
# def mlf(env):
#     ret_act = None
#     max_score = -1
#     for i in xrange(env.job_count):
#         for j in xrange(env.mac_count):
#             act = Action(env.jobs[i].id, env.macs[j].id)
#             if env.check_act(act):
#                 score = 0
#                 for k in xrange(env.pa.res_num):
#                     res_used = (env.macs[j].state[k] > 0)
#                     score += res_used.sum()
#
#                 if score > max_score:
#                     max_score = score
#                     ret_act = act
#     return ret_act
#
# def llf(env):
#     ret_act = None
#     min_score = sys.maxint
#     for i in xrange(env.job_count):
#         for j in xrange(env.mac_count):
#             act = Action(env.jobs[i].id, env.macs[j].id)
#             if env.check_act(act):
#                 score = 0
#                 for k in xrange(env.pa.res_num):
#                     res_used = (env.macs[j].state[k] > 0)
#                     score += res_used.sum()
#
#                 if score < min_score:
#                     min_score = score
#                     ret_act = act
#     return ret_act

def fifo(env):
    if env.job_count == 0: return  None
    for j in xrange(env.mac_count):
        act = Action(env.jobs[0].id, env.macs[j].id)
        if env.check_act(act):
            return act
    return None


def sjf(env):
    if env.job_count == 0: return  None
    min_duration = sys.maxint
    k = 0
    for i in xrange(env.job_count):
        if env.jobs[i].duration < min_duration:
            min_duration = env.jobs[i].duration
            k = i
    for j in xrange(env.mac_count):
        act = Action(env.jobs[k].id, env.macs[j].id)
        if env.check_act(act):
            return act

    return None

def mlf(env):
    if env.job_count == 0: return  None
    ret_act = None
    max_score = -1
    for j in xrange(env.mac_count):
        act = Action(env.jobs[0].id, env.macs[j].id)
        if env.check_act(act):
            score = 0
            for k in xrange(env.pa.res_num):
                res_used = (env.macs[j].state[k] > 0)
                score += res_used.sum()

            if score > max_score:
                max_score = score
                ret_act = act
    return ret_act

def llf(env):
    if env.job_count == 0: return  None
    ret_act = None
    min_score = sys.maxint
    for j in xrange(env.mac_count):
        act = Action(env.jobs[0].id, env.macs[j].id)
        if env.check_act(act):
            score = 0
            for k in xrange(env.pa.res_num):
                res_used = (env.macs[j].state[k] > 0)
                score += res_used.sum()

            if score < min_score:
                min_score = score
                ret_act = act
    return ret_act


def run(env, act_id):
    if act_id == 0:
        return fifo(env)
    elif act_id == 1:
        return sjf(env)
    elif act_id == 2:
        return mlf(env)
    elif act_id == 3:
        return llf(env)
    else:
        return None


