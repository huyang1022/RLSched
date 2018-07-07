from environment import Environment
from parameter import Parameter
from mac_generator import MacGenerator
from job_generator import JobGenerator
import act_generator
from element import Machine, Job
import tensorflow as tf
import numpy as np
from actor_critic import Actor, Critic
import os
import matplotlib.pyplot as plt

LOG_DIR = "./log"
LOG_FILE = "log_rl"
MODEL_DIR = "./model"

if __name__ == '__main__':
    sess = tf.Session()
    pa = Parameter()
    actor = Actor(sess, pa)
    critic = Critic(sess, pa)
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    saver = tf.train.Saver()
    logger = open(LOG_FILE, "w")  # file to record the logs

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    Machine.reset()
    Job.reset()
    env = Environment(pa)
    mac_gen = MacGenerator(pa)
    job_gen = JobGenerator(pa)
    env.job_gen = job_gen
    env.mac_gen = mac_gen
    plt_d = []


    for i in xrange(pa.exp_epochs):
        ep_s, ep_a, ep_r, ep_v, ep_w, ep_a_n= [], [], [], [], [], []
        print "================", "Start EP", i, "================"
        for j in xrange(pa.batch_num):
            env.reset()
            env.add_cluster()
            env.batch_id = j

            buffer_s, buffer_a, buffer_r, buffer_v, butter_w, butter_a_n = [], [], [], [], [], []
            td_sum = 0.0
            td_num = 0.0
            while True:
                state = env.obs()
                if i < pa.su_epochs:
                    act_id = act_generator.get_id(env)
                else:
                    act_id = actor.predict(state[np.newaxis, :])
                state_, reward, done, info, info1 = env.step_act(act_id)

                buffer_s.append(state)
                buffer_a.append(act_id)
                buffer_r.append(reward)
                butter_w.append(info)
                butter_a_n.append(info1)

                if done or env.current_time > pa.exp_len:

                    value = 0.0
                    for r in buffer_r[::-1]:
                        value = r + pa.discount_rate * value
                        buffer_v.append(value)
                    buffer_v.reverse()

                    buffer_s, buffer_a,  buffer_v = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v)
                    break

            ep_s.append(buffer_s)
            ep_a.append(buffer_a)
            ep_v.append(buffer_v)
            ep_r.extend(buffer_r)
            ep_w.extend(butter_w)
            ep_a_n.extend(butter_a_n)

        print "================", "Train EP", i, "================"
        ep_td, ep_c_loss, ep_a_loss = [], [], []
        for j in xrange(pa.batch_num):
            td_error, critic_loss = critic.learn(ep_s[j], ep_v[j])
            if i < pa.su_epochs:
                actor_loss = actor.su_train(ep_s[j], ep_a[j])
            else:
                actor_loss = actor.learn(ep_s[j], ep_a[j], td_error)
            ep_td.append(td_error)
            ep_c_loss.append(critic_loss)
            ep_a_loss.append(actor_loss)

        ep_td = np.concatenate(ep_td)
        ep_a = np.concatenate(ep_a)
        ep_c_loss = np.array(ep_c_loss)
        ep_a_loss = np.array(ep_a_loss)

        plt_d.append((np.sum(ep_w) + job_gen.total_len) * 1.0 / pa.job_num / pa.batch_num)

        print \
            "EP:", i, "\n", \
            "Batch Number:", pa.batch_num, "\n", \
            "Actions: ", np.sum(ep_a_n), "\n", \
            "EP_avg_c_loss: ", np.mean(ep_c_loss), "\n", \
            "EP_avg_a_loss: ", np.mean(ep_a_loss), "\n", \
            "EP_avg_td_error: ", np.mean(ep_td), "\n", \
            "EP_mean_reward: ", np.mean(ep_r), "\n", \
            "EP_batch_reward: ", np.sum(ep_r) / pa.batch_num, "\n", \
            "EP_avg_job_duration: ", plt_d[-1], "\n"

        logger.write("EP: %d\n" % i)
        logger.write("Batch Number: %d\n" % pa.batch_num)
        logger.write("Actions: %d\n" % np.sum(ep_a_n))
        logger.write("EP_avg_c_loss: %f\n" % np.mean(ep_c_loss))
        logger.write("EP_avg_a_loss: %f\n" % np.mean(ep_a_loss))
        logger.write("EP_avg_td_error: %f\n" % (np.mean(ep_td)))
        logger.write("EP_mean_reward: %f\n" % np.mean(ep_r))
        logger.write("EP_batch_reward: %f\n" % (np.sum(ep_r) / pa.batch_num))
        logger.write("EP_avg_job_duration: %f\n\n" % plt_d[-1])
        logger.flush()


        if i % pa.save_step == 0:
            saver.save(sess, "%s/%d.ckpt" % (MODEL_DIR, i))
    logger.write(str(plt_d))
    logger.flush()
    plt.plot(np.arange(len(plt_d)), plt_d)
    plt.show()

