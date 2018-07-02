import tensorflow as tf
import numpy as np
from parameter import Parameter


class Actor(object):
    def __init__(self, sess, pa):
        # type: (tf.Session, Parameter) -> None
        self.sess =sess
        self.pa = pa
        self.s_shape = [None, (pa.mac_num + pa.job_queue_num) * pa.res_num, pa.res_slot]
        self.a_num = pa.alg_num
        self.l_r = pa.learn_rate

        with tf.variable_scope("Actor"):
            with tf.variable_scope("Input"):
                self.state = tf.placeholder(tf.float32, self.s_shape, name = "state")
                self.act = tf.placeholder(tf.int32, [None, 1], name = "act")
                self.td_error = tf.placeholder(tf.float32, [None, 1], name = "td_error")

            with tf.variable_scope("Net"):
                l1 = tf.layers.dense(self.state, 600, name = "hidden_layer1")
                l2 = tf.layers.dense(l1, 100, name = "hidden_layer2")
                out = tf.layers.dense(l2, self.a_num, tf.nn.softmax, name = "act_prob")

                self.act_prob = out
                self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Actor/Net")

            with tf.variable_scope("Loss"):
                log_prob = tf.multiply(tf.log(self.act_prob), tf.squeeze(tf.one_hot(self.act, self.a_num)))
                sum_prob  = tf.reduce_sum(log_prob, axis=1, keepdims=True) * self.td_error
                self.loss = -sum_prob

            with tf.variable_scope('Train'):
                self.gradients = tf.gradients(self.loss, self.parameters)
                opt = tf.train.RMSPropOptimizer(self.l_r, name='RMSProp')
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
        ret_act = np.random.choice(np.arange(self.a_num), p=ret_prob.ravel())
        return ret_act

class Critic(object):
    def __init__(self, sess, pa):
        # type: (tf.Session, Parameter) -> None
        self.sess = sess
        self.pa = pa
        self.s_shape = [None, (pa.mac_num + pa.job_queue_num) * pa.res_num, pa.res_slot]
        self.l_r = pa.learn_rate

        with tf.variable_scope("Critic"):
            with tf.variable_scope("Input"):
                self.state = tf.placeholder(tf.float32, [None, self.s_shape], name = "state")
                self.value = tf.placeholder(tf.float32, [None, 1], name = "value")

            with tf.variable_scope("Net"):
                l1 = tf.layers.dense(self.state, 600, name = "hidden_layer1")
                l2 = tf.layers.dense(l1, 100, name = "hidden_layer2")
                out = tf.layers.dense(l2, 1,  name = "value")

                self.value_eval = out
                self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Critic/Net")

            with tf.variable_scope("Loss"):
                self.td_error = tf.subtract(self.value, self.value_eval)
                self.loss = tf.reduce_mean(tf.square(self.td_error))


            with tf.variable_scope('Train'):
                self.gradients = tf.gradients(self.loss, self.parameters)
                opt = tf.train.RMSPropOptimizer(self.l_r, name='RMSProp')
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

