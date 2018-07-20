from environment import Environment
from parameter import Parameter
from mac_generator import MacGenerator
from job_generator import JobGenerator
import act_generator
import tensorflow as tf
import numpy as np
from actor_critic import Actor, Critic
import os
# import plot
import multiprocessing as mp
import time

LOG_DIR = "./log"
LOG_FILE = LOG_DIR + "/rl_log"
MODEL_DIR = "./model"

def master(pa, net_queues, exp_queues):
    sess = tf.Session()
    actor = Actor(sess, pa)
    critic = Critic(sess, pa)
    sess.run(tf.global_variables_initializer())
    s_time = time.time()

    # writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    # saver = tf.train.Saver()
    logger = open(LOG_FILE, "w")  # file to record the logs
    plt_avg_data = []
    plt_min_data = []
    plt_max_data = []
    plt_test_data = []

    for i in xrange(pa.exp_epochs):
        print "================", "Start EP", i, "================"
        a_parameters = actor.get_parameters()
        c_parameters = critic.get_parameters()
        for j in xrange(pa.batch_num):
            net_queues[j].put([a_parameters, c_parameters])
            if pa.test_flag:
                net_queues[j + pa.batch_num].put([a_parameters, c_parameters])

        ep_s, ep_a, ep_v, ep_train_w, ep_test_w = [], [], [], [], []
        for j in xrange(pa.batch_num):
            buffer_s, buffer_a, buffer_v, butter_w = exp_queues[j].get()
            ep_s.append(buffer_s)
            ep_a.append(buffer_a)
            ep_v.append(buffer_v)
            ep_train_w.append(np.sum(butter_w))

            if pa.test_flag:
                butter_test_w = exp_queues[j + pa.batch_num].get()
                ep_test_w.append(np.sum(butter_test_w))

        print "================", "Train EP", i, "================"
        ep_td, ep_c_loss, ep_a_loss, ep_a_entropy = [], [], [], []
        # ep_a_gradients, ep_c_gradients =[], []
        for j in xrange(pa.batch_num):
            td_error, c_loss = critic.learn(ep_s[j], ep_v[j])
            # td_error, c_loss, c_gradients = critic.get_gradients(ep_s[j], ep_v[j])
            if i < pa.su_epochs:
                a_entropy, a_loss = actor.s_learn(ep_s[j], ep_a[j])
                # a_entropy, a_loss, a_gradients = actor.get_s_gradients(ep_s[j], ep_a[j])
            else:
                a_entropy, a_loss = actor.learn(ep_s[j], ep_a[j], td_error)
                # a_entropy, a_loss, a_gradients = actor.get_gradients(ep_s[j], ep_a[j], td_error)
            # ep_a_gradients.append(a_gradients)
            # ep_c_gradients.append(c_gradients)
            ep_td.append(td_error)
            ep_c_loss.append(c_loss)
            ep_a_loss.append(a_loss)
            ep_a_entropy.append(a_entropy)

        # for j in xrange(pa.batch_num):
        #     actor.update_parameters(ep_a_gradients[j])
        #     critic.update_parameters(ep_c_gradients[j])

        ep_td = np.concatenate(ep_td)
        ep_c_loss = np.array(ep_c_loss)
        ep_a_loss = np.array(ep_a_loss)
        ep_a_entropy = np.array(ep_a_entropy)

        plt_avg_data.append(float(np.mean(ep_train_w)))
        plt_min_data.append(np.min(ep_train_w))
        plt_max_data.append(np.max(ep_train_w))
        if pa.test_flag:
            plt_test_data.append(float(np.mean(ep_test_w)))
        else:
            plt_test_data.append(0.0)


        print \
            "EP:", i, "\n", \
            "Batch Number:", pa.batch_num, "\n", \
            "EP_avg_c_loss: ", np.mean(ep_c_loss), "\n", \
            "EP_avg_a_loss: ", np.mean(ep_a_loss), "\n", \
            "EP_avg_a_entropy: ", np.mean(ep_a_entropy), "\n", \
            "EP_avg_td_error: ", np.mean(ep_td), "\n", \
            "EP_train_time: ", time.time() - s_time, "\n", \
            "EP_avg_makespan: ", plt_avg_data[-1], "\n", \
            "EP_min_makespan: ", plt_min_data[-1], "\n", \
            "EP_max_makespan: ", plt_max_data[-1], "\n", \
            "EP_test_makespan: ", plt_test_data[-1], "\n"

        logger.write("EP: %d\n" % i)
        logger.write("Batch Number: %d\n" % pa.batch_num)
        logger.write("EP_avg_c_loss: %f\n" % np.mean(ep_c_loss))
        logger.write("EP_avg_a_loss: %f\n" % np.mean(ep_a_loss))
        logger.write("EP_avg_a_entropy: %f\n" % np.mean(ep_a_entropy))
        logger.write("EP_avg_td_error: %f\n" % (np.mean(ep_td)))
        logger.write("EP_train_time: %f\n" % (time.time() - s_time))
        logger.write("EP_avg_makespan: %f\n" % plt_avg_data[-1])
        logger.write("EP_min_makespan: %f\n" % plt_min_data[-1])
        logger.write("EP_max_makespan: %f\n" % plt_max_data[-1])
        logger.write("EP_test_makespan: %f\n\n" % plt_test_data[-1])
        logger.flush()

    # plot.run()


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
                    value = - pa.job_num
                for r in buffer_r[::-1]:
                    value = r + pa.discount_rate * value
                    buffer_v.append(value)
                buffer_v.reverse()

                buffer_s, buffer_a, buffer_v = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v)
                exp_queue.put([buffer_s, buffer_a, buffer_v, butter_w])
                break
            state = state_



def tester(batch_id, pa, net_queue, exp_queue):
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
        butter_w = []
        while True:
            if i < pa.su_epochs:
                act_id = act_generator.get_id(env, i)
            else:
                act_id = actor.predict(state[np.newaxis, :])
            state_, reward, done, info = env.step_act(act_id)
            butter_w.append(info)

            if done or env.current_time >= pa.exp_len:
                exp_queue.put(butter_w)
                break
            state = state_

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
        if pa.test_flag:
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

    if pa.test_flag:
        testers = []
        for i in xrange(pa.batch_num):
            testers.append(mp.Process(target=tester,
                                  args=(pa.batch_num + i, pa, net_queues[pa.batch_num + i], exp_queues[pa.batch_num + i])))
        for i in xrange(pa.batch_num):
            testers[i].start()



    coordinator.join()


if __name__ == '__main__':
    main()
