from environment import Environment
from parameter import Parameter
from mac_generator import MacGenerator
from job_generator import JobGenerator
from element import Machine, Job
import plot
import tensorflow as tf
import numpy as np
from actor_critic import Actor, Critic
import os

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


    for i in xrange(pa.exp_epochs):
        ep_s, ep_a, ep_r, ep_v, ep_w = [], [], [], [], []
        print "================", "Start EP", i, "================"
        for j in xrange(pa.batch_num):
            env.reset()
            env.add_cluster()
            env.batch_id = j

            buffer_s, buffer_a, buffer_r, buffer_v, butter_w = [], [], [], [], []
            td_sum = 0.0
            td_num = 0.0
            while True:
                state = env.obs()
                act_id = actor.predict(state[np.newaxis, :])
                state_, reward, done, info = env.step_act(act_id)

                buffer_s.append(state)
                buffer_a.append(act_id)
                buffer_r.append(reward)
                butter_w.append(info)

                if done or env.current_time > pa.exp_len:

                    value = 0
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

        print "================", "Train EP", i, "================"
        for j in xrange(pa.batch_num):
            td_error = critic.learn(ep_s[j], ep_v[j])
            actor.learn(ep_s[j], ep_v[j], td_error)
            td_sum += np.sum(td_error)
            td_num += np.size(td_error)


        print \
            "EP:", i, "\n", \
            "Batch Number:", pa.batch_num, "\n", \
            "EP_mean_reward: ", np.mean(ep_r), "\n", \
            "EP_max_reward: ", np.max(ep_r), "\n", \
            "EP_avg_reward: ", np.sum(ep_r) / pa.batch_num, "\n", \
            "EP_avg_td_error: ", td_sum / td_num, "\n", \
            "EP_avg_duration: ", (np.sum(ep_w) + job_gen.total_len) * 1.0 / pa.job_num / pa.batch_num, "\n"

        logger.write("EP: %d\n" % i)
        logger.write("Batch Number: %d\n" % pa.batch_num)
        logger.write("EP_mean_reward: %f\n" % np.mean(ep_r))
        logger.write("EP_max_reward: %f\n" % np.max(ep_r))
        logger.write("EP_avg_reward: %f\n" % (np.sum(ep_r) / pa.batch_num))
        logger.write("EP_avg_td_error: %f\n" % (td_sum / td_num))
        logger.write("EP_avg_duration: %f\n\n" % ((np.sum(ep_w) + job_gen.total_len) * 1.0 / pa.job_num / pa.batch_num))
        logger.flush()


        if i % pa.save_step == 0:
            saver.save(sess, "%s/%d.ckpt" % (MODEL_DIR, i))



