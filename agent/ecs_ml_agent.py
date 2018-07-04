from environment import Environment
from element import Action

class Edge(object):
    def __init__(self, u, v, f, c):
        self.u = u
        self.v = v
        self.f = f
        self.c = c

def add_edge(u, v, f, c, edges, links):
    edges.append(Edge(u, v, f, c))
    links[u].append(len(edges) - 1)
    edges.append(Edge(v, u, 0, -c))
    links[v].append(len(edges) - 1)



def spfa(source, sink, edges, links, path):
    v_queue = []
    v_set = set()

    v_queue.append(source)
    v_set.add(source)

    d = []
    for i in xrange(sink + 1):
        d.append(float('inf'))

    path[source] = -1
    d[source] = 0

    for i in v_queue:
        v_set.remove(i)
        for j in links[i]:
            if (edges[j].f > 0) and (d[i] + edges[j].c < d[edges[j].v] - 0.00000001):
                d[edges[j].v] = d[i] + edges[j].c
                path[edges[j].v] = j
                if edges[j].v not in v_set:
                    v_set.add(edges[j].v)
                    v_queue.append(edges[j].v)

    if d[sink] < float('inf'):
        return True
    else:
        return False


def run(env):
    # type: (Environment) -> object

    edges = []
    path = []
    links = {}
    mincost = 0
    maxflow = 0

    job_set = set()
    mac_num = env.mac_count
    job_num = min(env.job_count, env.pa.ecs_process_num)
    source = 0
    sink = mac_num + job_num + 1


    for i in xrange(sink + 1):
        links[i] = []
        path.append(-1)

    for i in xrange(job_num):
        add_edge(source, i + 1, 1, 0, edges, links)
    for i in xrange(mac_num):
        add_edge(job_num + i + 1, sink, 1, 0, edges, links)

    for i in xrange(job_num):
        for j in xrange(mac_num):
            score = 0
            for k in xrange(env.pa.res_num):
                res_avail = (env.macs[j].state[k] == 0)
                if res_avail.sum() < env.jobs[i].res_vec[k]:
                    score = 0
                    break
                score -= env.jobs[i].res_vec[k] * 1.0 / res_avail.sum()

            if score < 0:
                job_set.add(i)
                add_edge(i + 1, job_num + j + 1, 1, score, edges, links)


    while spfa(source, sink, edges, links, path):
        flow = float('inf')

        i = path[sink]
        while i != -1:
            if flow > edges[i].f:
                flow = edges[i].f
            i = path[edges[i].u]


        i = path[sink]
        while i != -1:
            edges[i].f -= flow
            edges[i^1].f += flow
            mincost += edges[i].c * flow
            i = path[edges[i].u]

        maxflow += flow

    # print maxflow, mincost

    act_list = []
    pop_list = []
    for i in xrange(job_num):
        if i not in job_set:
            pop_list.append(env.jobs[i])
        else:
            for j in links[i + 1]:
                if edges[j].f == 0 and edges[j].c < 0:
                    act = Action(env.jobs[i].id, env.macs[edges[j].v - job_num - 1].id)
                    act.show()
                    env.take_act(act)
                    act_list.append(act)

    for i in act_list:
        env.pop_job(i.job_id)

    if env.pa.sched_flag:
        for i in pop_list:
            env.pop_job(i.id)
            env.add_job(i)

    return len(act_list)

def schedule(env):
    # type: (Environment) -> None
    for i in xrange(env.pa.ecs_sched_num):
        # run(env)
        if not run(env): break

#
# if __name__ == "__main__":
#     schedule()


