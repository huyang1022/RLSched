import tensorflow as tf
import numpy as np
from parameter import Parameter
import os
import shutil
import gym

env = gym.make("CartPole-v1")
N_S = env.observation_space.shape[0]
N_A = env.action_space.n
L_R = 0.001
GAMMA = 0.9
LOG_DIR = './log'
EPS = 1e-6

class Actor(object):
    def __init__(self, sess):
        self.sess =sess
        with tf.variable_scope("Actor"):
            with tf.variable_scope("Input"):
                self.state = tf.placeholder(tf.float32, [None, N_S], name = "state")
                self.act = tf.placeholder(tf.int32, [None, 1], name = "act")
                self.td_error = tf.placeholder(tf.float32, [None, 1], name = "td_error")

            with tf.variable_scope("Net"):
                l1 = tf.layers.dense(self.state, 200, tf.nn.relu , name = "hidden_layer1")
                out = tf.layers.dense(l1, N_A, tf.nn.softmax, name = "act_prob")

                self.act_prob = out
                self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Actor/Net")

            with tf.variable_scope("Loss"):
                log_prob = tf.multiply(tf.log(self.act_prob + EPS), tf.squeeze(tf.one_hot(self.act, N_A)))
                sum_prob  = tf.reduce_sum(tf.multiply(tf.reduce_sum(log_prob, axis=1, keepdims=True), - self.td_error))
                entropy = tf.reduce_sum(tf.multiply(self.act_prob, tf.log(self.act_prob + EPS)))
                self.loss = sum_prob + L_R * entropy

            with tf.variable_scope('Train'):
                self.gradients = tf.gradients(self.loss, self.parameters)
                opt = tf.train.RMSPropOptimizer(L_R, name='RMSProp')
                self.update = opt.apply_gradients(zip(self.gradients, self.parameters))

    def learn(self, state, act, td_error):
        feed_dict = {
            self.state: state,
            self.act: act,
            self.td_error: td_error
        }
        ret_loss , _ = self.sess.run([self.loss, self.update], feed_dict)
        return ret_loss

    def choose_action(self, state):
        feed_dict = {
            self.state: state
        }
        ret_prob = self.sess.run(self.act_prob, feed_dict)
        ret_act = np.random.choice(np.arange(N_A), p=ret_prob.ravel())
        return ret_act

class Critic(object):
    def __init__(self, sess):
        self.sess =sess
        with tf.variable_scope("Critic"):
            with tf.variable_scope("Input"):
                self.state = tf.placeholder(tf.float32, [None, N_S], name = "state")
                self.value = tf.placeholder(tf.float32, [None, 1], name = "value")

            with tf.variable_scope("Net"):
                l1 = tf.layers.dense(self.state, 200, tf.nn.relu, name = "hidden_layer1")
                out = tf.layers.dense(l1, 1,  name = "value")

                self.value_eval = out
                self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Critic/Net")

            with tf.variable_scope("Loss"):
                self.td_error = tf.subtract(self.value, self.value_eval)
                self.loss = tf.reduce_mean(tf.square(self.td_error))


            with tf.variable_scope('Train'):
                self.gradients = tf.gradients(self.loss, self.parameters)
                opt = tf.train.RMSPropOptimizer(L_R, name='RMSProp')
                self.update = opt.apply_gradients(zip(self.gradients, self.parameters))

    def learn(self, state, value):
        feed_dict = {
            self.state: state,
            self.value: value
        }
        ret_td , _ = self.sess.run([self.td_error, self.update], feed_dict)
        return ret_td

    def get_value(self, state):
        feed_dict = {
            self.state: state
        }
        value_eval = self.sess.run(self.value_eval, feed_dict)
        ret_value = value_eval[0,0]
        return ret_value


if __name__ == '__main__':
    sess = tf.Session()
    actor = Actor(sess)
    critic = Critic(sess)
    sess.run(tf.global_variables_initializer())
    # if os.path.exists(LOG_DIR):
        # shutil.rmtree(LOG_DIR)
    tf.summary.FileWriter(LOG_DIR, sess.graph)

    r_list = []
    for i in xrange(500):
        s = env.reset()


        ep_r = 0
        total_step = 1
        buffer_s, buffer_a, buffer_r,  buffer_v = [], [], [], []
        while True:
            # env.render()
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
