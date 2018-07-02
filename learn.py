from environment import Environment
from parameter import Parameter
from mac_generator import MacGenerator
from job_generator import JobGenerator
from element import Machine, Job
from agent import ecs_agent, ecs_dp_agent, ecs_ml_agent, swarm_agent, pack_agent, k8s_agent
import plot
import tensorflow as tf
import numpy as np
from actor_critic import Actor, Critic

def run(agent):

    Machine.reset()
    Job.reset()

    pa = Parameter()
    pa.agent = agent
    env = Environment(pa)

    mac_gen = MacGenerator(pa)
    job_gen = JobGenerator(pa)


    for i in mac_gen.mac_sequence:
        env.add_machine(i)

    job_idx = 0
    current_time = 0




    while True:


        env.step()
        env.time_log()

        while ( env.job_count < pa.job_queue_num and
                job_gen.job_sequence[job_idx] is not None and
                job_gen.job_sequence[job_idx].submission_time <= current_time
        ):
            if (job_gen.job_sequence[job_idx].submission_time == current_time):   #  add job to environment
                env.add_job(job_gen.job_sequence[job_idx])
            job_idx += 1

        if agent == "ecs":
            ecs_agent.schedule(env)
        elif agent =="k8s":
            k8s_agent.schedule(env)
        elif agent == "pack":
            pack_agent.schedule(env)
        elif agent == "ecs_dp":
            ecs_dp_agent.schedule(env)
        elif agent == "ecs_ml":
            ecs_ml_agent.schedule(env)
        elif agent == "swarm":
            swarm_agent.schedule(env)

        if job_gen.job_sequence[job_idx] is None:
            if env.status() == "Idle": # finish all jobs
                break
        current_time += 1

    env.job_log()
    env.finish()

def main():
    run("ecs_dp")
    run("ecs_ml")
    run("k8s")
    run("swarm")
    plot.run()
    # run("pack")


if __name__ == '__main__':
    sess = tf.Session()
    pa = Parameter()

    actor = Actor(sess, pa)
    critic = Critic(sess, pa)
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter("./log", sess.graph)

    Machine.reset()
    Job.reset()
    env = Environment(pa)
    mac_gen = MacGenerator(pa)
    job_gen = JobGenerator(pa)



    exp_n = 0

    while exp_n < pa.exp_num:
        exp_n += 1

        env.reset()
        for i in mac_gen.mac_sequence:
            env.add_machine(i)

        job_idx = 0
        current_time = 0

        ep_r = 0
        buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
        while True:
            env.step()
            state = env.obs()
            act_num = actor.choose_action(state[np.newaxis, :])

            while (env.job_count < pa.job_queue_num and
                   job_gen.job_sequence[job_idx] is not None and
                   job_gen.job_sequence[job_idx].submission_time <= current_time
            ):
                if (job_gen.job_sequence[job_idx].submission_time == current_time):  # add job to environment
                    env.add_job(job_gen.job_sequence[job_idx])
                job_idx += 1

            if job_gen.job_sequence[job_idx] is None:
                if env.status() == "Idle":  # finish all jobs
                    break
            current_time += 1



    r_list = []
    for i in xrange(500):
        s = env.reset()


        ep_r = 0
        total_step = 1
        buffer_s, buffer_a, buffer_r,  buffer_v = [], [], [], []
        while True:
            a = actor.choose_action(s[np.newaxis, :])
            s_, r, done, info = env.step(a)
            if done: r = -5
            ep_r += r
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)

            if total_step % 10 == 0 or done:  # update global and assign to local net
                if done:
                    v_s_ = 0  # terminal
                else:
                    v_s_ = critic.get_value(s_[np.newaxis, :])

                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    buffer_v.append(v_s_)
                buffer_v.reverse()

                buffer_s, buffer_a, buffer_v = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v)
                butter_td = critic.learn(buffer_s, buffer_v)
                actor.learn(buffer_s, buffer_a, butter_td)
                buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []

            s = s_
            total_step += 1

            if done:
                if len(r_list) == 0:  # record running episode reward
                    r_list.append(ep_r)
                else:
                    r_list.append(0.99 * r_list[-1] + 0.01 * ep_r)
                print(
                    "Ep:", i,
                    "| Ep_r: %i" % r_list[-1],
                    "| Ep_r: %i" % ep_r,
                )
                break
