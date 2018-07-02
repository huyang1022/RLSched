from environment import Environment
from element import Action

def schedule(env):
    # type: (Environment) -> object

    job_num = min(env.job_count, env.pa.sched_num)
    act_list = []
    pop_list = []

    for i in xrange(job_num):
        max_score = 0
        act = None
        for j in xrange(env.mac_count):
            score = 0
            for k in xrange(env.pa.res_num):
                res_avail = (env.macs[j].state[k] == 0)
                if res_avail.sum() < env.jobs[i].res_vec[k]:
                    score = 0
                    break
                score += res_avail.sum() * env.jobs[i].res_vec[k]

            if score > max_score:
                max_score = score
                act = Action(env.jobs[i].id, env.macs[j].id)

        if act is not None:
            act_list.append(act)
            act.show()
            env.take_act(act)
        else:
            pop_list.append(env.jobs[i])

    for i in act_list:
        env.pop_job(i.job_id)

    if env.pa.sched_flag:
        for i in pop_list:
            env.pop_job(i.id)
            env.add_job(i)

    return len(act_list)
