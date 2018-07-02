import numpy as np
import matplotlib.pyplot as plt


def read_data(agent):
    ret_list= list()
    in_file = open("data/%s" % agent, "r")
    while True:
        line = in_file.readline()
        line = line.split()
        if not line: break
        ret_list.append(float(line[1]))

    return ret_list

def read_log(agent):
    in_file = open("log/%s" % agent, "r")
    num = 0
    cpu_usage = 0.0
    mem_usage = 0.0
    pend_num = 0
    pend_cpu = 0.0
    pend_mem = 0.0
    max_pend = 0

    while True:
        line = in_file.readline()
        line = line.split()
        if not line: break
        cpu = float(line[11])
        mem = float(line[13])
        if cpu > 0:
            num += 1
            cpu_usage += cpu
            mem_usage += mem
            if int(line[5]) > 0:
                max_pend = max(max_pend, int(line[5]))
                pend_num += 1
                pend_cpu += cpu
                pend_mem += mem

    print "Agent: %s,  Cpu: %f,  Mem: %f" % (agent, cpu_usage/num, mem_usage/num)
    if pend_num>0:
        print "Agent: %s,  Pend Cpu: %f,  Pend Mem: %f,  Max num: %d" % (agent, pend_cpu/pend_num, pend_mem/pend_num, max_pend)

def cmp_agent(agent1, agent2):
    l = list()
    sum_ratio = 0
    max_ratio = 0
    min_ratio = 1
    sum_a = 0.0
    sum_b = 0.0
    for i in xrange(len(agent1)):
        a = agent1[i]
        b = agent2[i]
        ratio = b * 1.0 / a
        # ratio = 1 - a * 1.0 / b
        l.append(ratio)
        sum_a += a
        sum_b += b
        sum_ratio += ratio
        max_ratio = max(ratio, max_ratio)
        min_ratio = min(ratio, min_ratio)

    l.sort()
    print "Total: ", len(l)
    print "Reduction: ", len([x for x in l if x > 1])
    print "Increment: ", len([x for x in l if x < 1])
    print "Maximum Improve: ", max_ratio
    print "Minimum Improve: ", min_ratio
    print "Average Ratio Improve: ", sum_ratio / len(l)
    print "Average Duration Improve: ", sum_b / sum_a
    return l

def plot_data(l):
    dx = 0.01
    sx = 0.5
    fx = 3
    i = 0
    y =[]
    while sx + dx * i < fx:
        y.append(len([x for x in l if sx + dx * i <= x < sx + dx * (i + 1)]))
        i += 1


    y = np.array(y)
    x = np.arange(sx,fx,dx)

    plt.plot(x, y.cumsum())

def run():

    agent_data = dict()
    agent = ["ecs_dp", "ecs_ml", "swarm", "k8s"]
    for i in agent:
        agent_data[i] = read_data(i)
        read_log(i)

    print "=============  ecs_dp vs. k8s  ================"
    l = cmp_agent(agent_data["ecs_dp"], agent_data["k8s"])
    plt.subplot(2, 1, 1)
    plot_data(l)
    print "=============  ecs_ml vs. k8s  ================"
    l = cmp_agent(agent_data["ecs_ml"], agent_data["k8s"])
    plot_data(l)
    plt.legend(["dp-k8s","ml-k8s"])
    print "=============  ecs_dp vs. swarm  ================"
    l = cmp_agent(agent_data["ecs_dp"], agent_data["swarm"])
    plt.subplot(2, 1, 2)
    plot_data(l)
    print "=============  ecs_ml vs. swarm  ================"
    l = cmp_agent(agent_data["ecs_ml"], agent_data["swarm"])
    plot_data(l)
    plt.legend(["dp-swarm","ml-swarm"])
    plt.show()


if __name__ == "__main__":
    run()