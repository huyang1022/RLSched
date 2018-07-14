from environment import Environment
from parameter import Parameter
from mac_generator import MacGenerator
from job_generator import JobGenerator
import act_generator
import tensorflow as tf
import numpy as np
from actor_critic import Actor, Critic
import os
import plot
import multiprocessing as mp

LOG_DIR = "./log"
LOG_FILE = LOG_DIR + "/rl_log"
MODEL_DIR = "./model"

def master(pa, net_queues, exp_queues):
    sess = tf.Session()
    actor = Actor(sess, pa)
    critic = Critic(sess, pa)
    sess.run(tf.global_variables_initializer())

    # writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    # saver = tf.train.Saver()
    logger = open(LOG_FILE, "w")  # file to record the logs
    plt_data = []
    for i in xrange(pa.exp_epochs):
        print "================", "Start EP", i, "================"
        a_parameters = actor.get_parameters()
        c_parameters = critic.get_parameters()
        for j in xrange(pa.batch_num):
            net_queues[j].put([a_parameters, c_parameters])

        ep_s, ep_a, ep_r, ep_v, ep_w = [], [], [], [], []
        for j in xrange(pa.batch_num):
            buffer_s, buffer_a, buffer_r, buffer_v, butter_w = exp_queues[j].get()
            ep_s.append(buffer_s)
            ep_a.append(buffer_a)
            ep_v.append(buffer_v)
            ep_r.extend(buffer_r)
            ep_w.extend(butter_w)

        print "================", "Train EP", i, "================"
        ep_td, ep_c_loss, ep_a_loss, ep_a_entropy = [], [], [], []
        for j in xrange(pa.batch_num):
            td_error, critic_loss = critic.learn(ep_s[j], ep_v[j])
            if i < pa.su_epochs:
                actor_entropy, actor_loss = actor.s_train(ep_s[j], ep_a[j])
            else:
                actor_entropy, actor_loss = actor.learn(ep_s[j], ep_a[j], td_error)
            ep_td.append(td_error)
            ep_c_loss.append(critic_loss)
            ep_a_loss.append(actor_loss)
            ep_a_entropy.append(actor_entropy)

        ep_td = np.concatenate(ep_td)
        ep_c_loss = np.array(ep_c_loss)
        ep_a_loss = np.array(ep_a_loss)
        ep_a_entropy = np.array(ep_a_entropy)

        plt_data.append(np.sum(ep_w) *1.0 /pa.batch_num)

        print \
            "EP:", i, "\n", \
            "Batch Number:", pa.batch_num, "\n", \
            "EP_avg_c_loss: ", np.mean(ep_c_loss), "\n", \
            "EP_avg_a_loss: ", np.mean(ep_a_loss), "\n", \
            "EP_avg_a_entropy: ", np.mean(ep_a_entropy), "\n", \
            "EP_avg_td_error: ", np.mean(ep_td), "\n", \
            "EP_mean_reward: ", np.mean(ep_r), "\n", \
            "EP_batch_reward: ", np.sum(ep_r) / pa.batch_num, "\n", \
            "EP_avg_makespan: ", plt_data[-1], "\n"

        logger.write("EP: %d\n" % i)
        logger.write("Batch Number: %d\n" % pa.batch_num)
        logger.write("EP_avg_c_loss: %f\n" % np.mean(ep_c_loss))
        logger.write("EP_avg_a_loss: %f\n" % np.mean(ep_a_loss))
        logger.write("EP_avg_a_entropy: %f\n" % np.mean(ep_a_entropy))
        logger.write("EP_avg_td_error: %f\n" % (np.mean(ep_td)))
        logger.write("EP_mean_reward: %f\n" % np.mean(ep_r))
        logger.write("EP_batch_reward: %f\n" % (np.sum(ep_r) / pa.batch_num))
        logger.write("EP_avg_jmakespan: %f\n\n" % plt_data[-1])
        logger.flush()

    plot.run()


def worker(batch_id, pa, net_queue, exp_queue):
    sess = tf.Session()
    actor = Actor(sess, pa)
    critic = Critic(sess, pa)

    env = Environment(pa)
    mac_gen = MacGenerator(pa)
    job_gen = JobGenerator(pa)
    env.job_gen = job_gen
    env.mac_gen = mac_gen


    for i in xrange(pa.exp_epochs):
        a_parameters, c_parameters = net_queue.get()
        actor.set_parameters(a_parameters)
        critic.set_parameters(c_parameters)
        env.reset()
        env.add_cluster()
        env.batch_id = batch_id
        state = env.obs()
        buffer_s, buffer_a, buffer_r, buffer_v, butter_w = [], [], [], [], []
        while True:
            if i < pa.su_epochs:
                act_id = act_generator.get_id(env, i)
            else:
                act_id = actor.predict(state[np.newaxis, :])
            state_, reward, done, info = env.step_act(act_id)

            buffer_s.append(state)
            buffer_a.append(act_id)
            buffer_r.append(reward)
            butter_w.append(info)

            if done or env.current_time >= pa.exp_len:
                if done:
                    value = 0
                else:
                    value = len(env.finished_jobs) - pa.job_num
                for r in buffer_r[::-1]:
                    value = r + pa.discount_rate * value
                    buffer_v.append(value)
                buffer_v.reverse()

                buffer_s, buffer_a, buffer_v = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v)
                break
            state = state_

        exp_queue.put([buffer_s, buffer_a, buffer_r, buffer_v, butter_w])



def main():
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    pa = Parameter()

    net_queues = []
    exp_queues = []
    for i in xrange(pa.batch_num):
        net_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    coordinator = mp.Process(target=master,
                             args=(pa, net_queues, exp_queues))
    coordinator.start()

    workers = []
    for i in xrange(pa.batch_num):
        workers.append(mp.Process(target=worker,
                                  args=(i, pa, net_queues[i], exp_queues[i])))

    for i in xrange(pa.batch_num):
        workers[i].start()

    coordinator.join()


if __name__ == '__main__':
    main()
