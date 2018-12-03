import tensorflow as tf
import numpy as np
from parameter import Parameter


class Actor(object):
    def __init__(self, sess, pa):
        # type: (tf.Session, Parameter) -> None
        self.sess =sess
        self.pa = pa
        m_num = pa.mac_train_num * pa.res_num * pa.mac_max_slot * pa.job_max_len
        # j_num = pa.job_train_num * pa.res_num * pa.job_max_slot * pa.job_max_len
        j_num = pa.job_train_num * (pa.res_num * pa.job_max_slot + pa.job_max_len)
        d_num = pa.job_train_num * pa.dag_max_depth * pa.job_max_len * 2
        self.s_dim = m_num + j_num + d_num
        self.a_dim =pa.mac_train_num * pa.job_train_num + 1
        self.l_r = pa.a_learn_rate

        with tf.variable_scope("Actor"):
            with tf.variable_scope("Input"):
                self.state = tf.placeholder(tf.float32, [None, self.s_dim], name = "state")
                self.act = tf.placeholder(tf.int32, [None, 1], name = "act")
                self.td_error = tf.placeholder(tf.float32, [None, 1], name = "td_error")


            with tf.variable_scope("Net"):
                # l_m = tf.reshape(self.state[:, :m_num], [-1, pa.mac_train_num * pa.res_num * pa.mac_max_slot, pa.job_max_len, 1])
                # l_j = tf.reshape(self.state[:, m_num:j_num+m_num], [-1, pa.job_train_num * pa.res_num * pa.job_max_slot, pa.job_max_len, 1])
                # l_d = tf.reshape(self.state[:, j_num+m_num:], [-1,  pa.job_train_num * pa.dag_max_depth, pa.job_max_len, 1])
                # c_m = tf.layers.conv2d(l_m, filters=32, kernel_size=[pa.mac_max_slot, pa.job_max_len], strides=[pa.mac_max_slot, pa.job_max_len], activation=tf.nn.relu6)
                # c_j = tf.layers.conv2d(l_j, filters=32, kernel_size=[pa.job_max_slot, pa.job_max_len], strides=[pa.job_max_slot, pa.job_max_len], activation=tf.nn.relu6)
                # c_d = tf.layers.conv2d(l_d, filters=32, kernel_size=[pa.dag_max_depth, pa.job_max_len], strides=[pa.dag_max_depth, pa.job_max_len], activation=tf.nn.relu6)
                # f_m = tf.reshape(c_m, [-1, pa.mac_train_num * pa.res_num * 32])
                # f_j = tf.reshape(c_j, [-1, pa.job_train_num * pa.res_num * 32])
                # f_d = tf.reshape(c_d, [-1, pa.job_train_num * 32])
                # l_con = tf.concat([f_m, f_j, f_d], 1)
                l1 = tf.layers.dense(self.state, 256, tf.nn.relu6, name = "hidden_layer1")
                # l2 = tf.layers.dense(l1, 256, tf.nn.relu6, name = "hidden_layer2")
                out = tf.layers.dense(l1, self.a_dim, tf.nn.softmax, name = "act_prob")

                self.act_prob = out
                self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Actor/Net")

            with tf.variable_scope("Loss"):
                # log_prob = tf.multiply(tf.log(self.act_prob + self.pa.eps), tf.squeeze(tf.one_hot(self.act, self.a_dim)))
                # sum_prob = tf.reduce_sum(tf.multiply(tf.reduce_sum(log_prob, axis=1, keepdims=True), - self.td_error))
                log_prob = tf.reduce_sum(tf.multiply(self.act_prob, tf.squeeze(tf.one_hot(self.act, self.a_dim))), axis=1, keepdims=True)
                sum_prob = tf.reduce_sum(tf.multiply(tf.log(log_prob), - self.td_error))
                self.entropy =  tf.reduce_sum(tf.multiply(self.act_prob, tf.log(self.act_prob + self.pa.eps)))
                self.loss = sum_prob + self.pa.entropy_rate * self.entropy



            with tf.variable_scope('Train'):
                # op = tf.train.AdamOptimizer(self.l_r, name="Adam")
                op = tf.train.RMSPropOptimizer(self.l_r, name='RMSProp')
                self.gradients = tf.gradients(self.loss, self.parameters)
                self.update = op.apply_gradients(zip(self.gradients, self.parameters))

            with tf.variable_scope("S_Loss"):
                s_error = tf.subtract(self.act_prob, tf.squeeze(tf.one_hot(self.act, self.a_dim)))
                self.s_entropy =  tf.reduce_sum(tf.multiply(self.act_prob, tf.log(self.act_prob + self.pa.eps)))
                self.s_loss = tf.reduce_sum(tf.square(s_error))

            with tf.variable_scope('S_Train'):
                s_op = tf.train.RMSPropOptimizer(self.l_r, name='RMSProp')
                self.s_gradients = tf.gradients(self.s_loss, self.parameters)
                self.s_update = s_op.apply_gradients(zip(self.s_gradients, self.parameters))

            self.input_parameters = []
            for param in self.parameters:
                self.input_parameters.append(
                    tf.placeholder(tf.float32, shape=param.get_shape()))

            self.set_ops = []
            for idx, param in enumerate(self.input_parameters):
                self.set_ops.append(self.parameters[idx].assign(param))

    def get_parameters(self):
        return self.sess.run(self.parameters)

    def set_parameters(self, parameters):
        self.sess.run(self.set_ops, feed_dict={
            i: d for i, d in zip(self.input_parameters, parameters)
        })

    def update_parameters(self, gradients):
        return self.sess.run(self.update, feed_dict={
            i: d for i, d in zip(self.gradients, gradients)
        })

    def get_gradients(self, state, act, td_error):
        feed_dict = {
            self.state: state,
            self.act: act,
            self.td_error: td_error
        }
        ret_entropy, ret_loss , ret_gradients = self.sess.run([self.entropy, self.loss, self.gradients], feed_dict)
        return ret_entropy, ret_loss, ret_gradients

    def get_s_gradients(self, state, act):
        feed_dict = {
            self.state: state,
            self.act: act,
        }
        ret_entropy, ret_loss , ret_gradients = self.sess.run([self.s_entropy, self.s_loss, self.s_gradients], feed_dict)
        return ret_entropy, ret_loss, ret_gradients


    def s_learn(self, state, act):
        feed_dict = {
            self.state: state,
            self.act: act
        }
        ret_entropy, ret_loss , _ = self.sess.run([self.s_entropy, self.s_loss, self.s_update], feed_dict)
        return ret_entropy, ret_loss

    def learn(self, state, act, td_error):
        feed_dict = {
            self.state: state,
            self.act: act,
            self.td_error: td_error
        }
        ret_entropy, ret_loss , _ = self.sess.run([self.entropy, self.loss, self.update], feed_dict)
        return ret_entropy, ret_loss

    def predict(self, state):
        feed_dict = {
            self.state: state
        }
        ret_prob = self.sess.run(self.act_prob, feed_dict)
        ret_act = np.random.choice(np.arange(self.a_dim), p=ret_prob.ravel())
        return ret_act

class Critic(object):
    def __init__(self, sess, pa):
        # type: (tf.Session, Parameter) -> None
        self.sess = sess
        self.pa = pa
        m_num = pa.mac_train_num * pa.res_num * pa.mac_max_slot * pa.job_max_len
        # j_num = pa.job_train_num * pa.res_num * pa.job_max_slot * pa.job_max_len
        j_num = pa.job_train_num * (pa.res_num * pa.job_max_slot + pa.job_max_len)
        d_num = pa.job_train_num * pa.dag_max_depth * pa.job_max_len * 2
        self.s_dim = m_num + j_num + d_num
        self.a_dim =pa.mac_train_num * pa.job_train_num + 1
        self.l_r = pa.c_learn_rate
        with tf.variable_scope("Critic"):
            with tf.variable_scope("Input"):
                self.state = tf.placeholder(tf.float32, [None, self.s_dim], name = "state")
                self.value_target = tf.placeholder(tf.float32, [None, 1], name = "value")

            with tf.variable_scope("Net"):
                # l_m = tf.reshape(self.state[:, :m_num], [-1, pa.mac_train_num * pa.res_num * pa.mac_max_slot, pa.job_max_len, 1])
                # l_j = tf.reshape(self.state[:, m_num:j_num+m_num], [-1, pa.job_train_num * pa.res_num * pa.job_max_slot, pa.job_max_len, 1])
                # l_d = tf.reshape(self.state[:, j_num+m_num:], [-1,  pa.job_train_num * pa.dag_max_depth, pa.job_max_len, 1])
                # c_m = tf.layers.conv2d(l_m, filters=32, kernel_size=[pa.mac_max_slot, pa.job_max_len], strides=[pa.mac_max_slot, pa.job_max_len], activation=tf.nn.relu6)
                # c_j = tf.layers.conv2d(l_j, filters=32, kernel_size=[pa.job_max_slot, pa.job_max_len], strides=[pa.job_max_slot, pa.job_max_len], activation=tf.nn.relu6)
                # c_d = tf.layers.conv2d(l_d, filters=32, kernel_size=[pa.dag_max_depth, pa.job_max_len], strides=[pa.dag_max_depth, pa.job_max_len], activation=tf.nn.relu6)
                # f_m = tf.reshape(c_m, [-1, pa.mac_train_num * pa.res_num * 32])
                # f_j = tf.reshape(c_j, [-1, pa.job_train_num * pa.res_num * 32])
                # f_d = tf.reshape(c_d, [-1, pa.job_train_num * 32])
                # l_con = tf.concat([f_m, f_j, f_d], 1)
                l1 = tf.layers.dense(self.state, 256, tf.nn.relu6, name = "hidden_layer1")
                # l2 = tf.layers.dense(l1, 256, tf.nn.relu6, name = "hidden_layer2")
                out = tf.layers.dense(l1, 1, name="value")

                self.value = out
                self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Critic/Net")

            with tf.variable_scope("Loss"):
                self.td_error = tf.subtract(self.value_target, self.value)
                self.loss = tf.reduce_mean(tf.square(self.td_error))


            with tf.variable_scope('Train'):
                # op = tf.train.AdamOptimizer(self.l_r, name="Adam")
                op = tf.train.RMSPropOptimizer(self.l_r, name='RMSProp')
                self.gradients = tf.gradients(self.loss, self.parameters)
                self.update = op.apply_gradients(zip(self.gradients, self.parameters))

            self.input_parameters = []
            for param in self.parameters:
                self.input_parameters.append(
                    tf.placeholder(tf.float32, shape=param.get_shape()))
            self.set_ops = []
            for idx, param in enumerate(self.input_parameters):
                self.set_ops.append(self.parameters[idx].assign(param))

    def get_parameters(self):
        return self.sess.run(self.parameters)

    def set_parameters(self, parameters):
        self.sess.run(self.set_ops, feed_dict={
            i: d for i, d in zip(self.input_parameters, parameters)
        })

    def update_parameters(self, gradients):
        return self.sess.run(self.update, feed_dict={
            i: d for i, d in zip(self.gradients, gradients)
        })

    def get_gradients(self, state, value):
        feed_dict = {
            self.state: state,
            self.value_target: value
        }
        ret_td , ret_loss, ret_gradients = self.sess.run([self.td_error, self.loss, self.gradients], feed_dict)
        return ret_td, ret_loss, ret_gradients



    def learn(self, state, value):
        feed_dict = {
            self.state: state,
            self.value_target: value
        }
        ret_td , ret_loss, _ = self.sess.run([self.td_error, self.loss, self.update], feed_dict)
        return ret_td, ret_loss

    def predict(self, state):
        feed_dict = {
            self.state: state
        }
        value = self.sess.run(self.value, feed_dict)
        ret_value = value[0,0]
        return ret_value

