import tensorflow as tf
import numpy as np
from parameter import Parameter


class Actor(object):
    def __init__(self, sess, pa):
        # type: (tf.Session, Parameter) -> None
        self.sess =sess
        self.pa = pa
        self.t_num = pa.mac_train_num + pa.job_train_num
        self.s_dim = self.t_num * pa.res_num * pa.res_slot
        self.a_dim =pa.mac_train_num * pa.job_train_num + 1
        self.l_r = pa.a_learn_rate

        with tf.variable_scope("Actor"):
            with tf.variable_scope("Input"):
                self.state = tf.placeholder(tf.float32, [None, self.s_dim], name = "state")
                self.act = tf.placeholder(tf.int32, [None, 1], name = "act")
                self.td_error = tf.placeholder(tf.float32, [None, 1], name = "td_error")

            with tf.variable_scope("Net"):
                l1_in = tf.reshape(self.state, [-1, self.t_num,  pa.res_num * pa.res_slot, 1])                            #   x, y,  1
                c1_v1 = tf.layers.conv2d(l1_in, filters=16, kernel_size=[1, 1], strides=[1, 1], activation=tf.nn.relu6)    #   x,  y ,  10
                c1_v2 = tf.layers.conv2d(c1_v1, filters=128, kernel_size=[1, 10], strides=[1, 10], activation=tf.nn.relu6)    #   x,  y ,  10
                # c1_v3 = tf.layers.conv2d(c1_v2, filters=128, kernel_size=[1, 2], strides=[1, 1], activation=tf.nn.relu6)    #   x,  y ,  10
                # l2_in = tf.reshape(self.state[:,-1:], [-1, 1, 1, 1])                            #   x, y,  1
                # c2_v1 = tf.layers.conv2d(l2_in, filters=16, kernel_size=[1, 1], strides=[1, 1], activation=tf.nn.relu6)   #   x, y/2,  32
                pool1 = tf.layers.max_pooling2d(c1_v2, pool_size = [1, 2], strides=[1,1])                                   #   x, y/10, 32
                flat1 = tf.reshape(pool1, [-1, self.t_num  * 128])
                # flat2 = tf.reshape(c2_v1, [-1, 16])
                # l_con = tf.concat([flat1, flat2], 1)
                l1 = tf.layers.dense(flat1, self.t_num * 128, tf.nn.relu, name = "hidden_layer1")
                # l2 = tf.layers.dense(l1, self.a_dim * 8, tf.nn.relu, name = "hidden_layer2")
                out = tf.layers.dense(l1, self.a_dim, tf.nn.softmax, name = "act_prob")

                self.act_prob = out
                self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Actor/Net")

            with tf.variable_scope("Loss"):
                log_prob = tf.multiply(tf.log(self.act_prob + self.pa.eps), tf.squeeze(tf.one_hot(self.act, self.a_dim)))
                sum_prob = - tf.reduce_sum(tf.multiply(tf.reduce_sum(log_prob, axis=1, keepdims=True), self.td_error))
                entropy =  tf.reduce_sum(tf.multiply(self.act_prob, tf.log(self.act_prob + self.pa.eps)))
                self.loss = sum_prob + self.pa.entropy_rate * entropy


            with tf.variable_scope('Train'):
                opt = tf.train.RMSPropOptimizer(self.l_r, name='RMSProp')
                self.gradients = tf.gradients(self.loss, self.parameters)
                self.update = opt.apply_gradients(zip(self.gradients, self.parameters))

            with tf.variable_scope("Su_Loss"):
                su_error = tf.subtract(self.act_prob, tf.squeeze(tf.one_hot(self.act, self.a_dim)))
                self.su_loss = tf.reduce_sum(tf.square(su_error))

            with tf.variable_scope('Su_Train'):
                su_opt = tf.train.RMSPropOptimizer(self.l_r, name='RMSProp')
                self.su_gradients = tf.gradients(self.su_loss, self.parameters)
                self.su_update = su_opt.apply_gradients(zip(self.su_gradients, self.parameters))

    def su_train(self, state, act):
        feed_dict = {
            self.state: state,
            self.act: act
        }
        ret_loss , _ = self.sess.run([self.su_loss, self.su_update], feed_dict)
        return ret_loss

    def learn(self, state, act, td_error):
        feed_dict = {
            self.state: state,
            self.act: act,
            self.td_error: td_error
        }
        ret_loss , _ = self.sess.run([self.loss, self.update], feed_dict)
        return ret_loss

    def predict(self, state):
        feed_dict = {
            self.state: state
        }
        ret_prob = self.sess.run(self.act_prob, feed_dict)
        # ret_cumsum = np.cumsum(ret_prob)
        # ret_act = (ret_cumsum > np.random.randint(1, 1000) / float(1000)).argmax()
        ret_act = np.random.choice(np.arange(self.a_dim), p=ret_prob.ravel())
        return ret_act

class Critic(object):
    def __init__(self, sess, pa):
        # type: (tf.Session, Parameter) -> None
        self.sess = sess
        self.pa = pa
        self.t_num = pa.mac_train_num + pa.job_train_num
        self.s_dim = self.t_num * pa.res_num * pa.res_slot
        self.a_dim =pa.mac_train_num * pa.job_train_num + 1
        self.l_r = pa.c_learn_rate
        with tf.variable_scope("Critic"):
            with tf.variable_scope("Input"):
                self.state = tf.placeholder(tf.float32, [None, self.s_dim], name = "state")
                self.value_target = tf.placeholder(tf.float32, [None, 1], name = "value")

            with tf.variable_scope("Net"):
                l1_in = tf.reshape(self.state, [-1, self.t_num,  pa.res_num * pa.res_slot, 1])                            #   x, y,  1
                c1_v1 = tf.layers.conv2d(l1_in, filters=16, kernel_size=[1, 1], strides=[1, 1], activation=tf.nn.relu6)    #   x,  y ,  10
                c1_v2 = tf.layers.conv2d(c1_v1, filters=128, kernel_size=[1, 10], strides=[1, 10], activation=tf.nn.relu6)    #   x,  y ,  10
                # c1_v3 = tf.layers.conv2d(c1_v2, filters=128, kernel_size=[1, 2], strides=[1, 1], activation=tf.nn.relu6)    #   x,  y ,  10
                # l2_in = tf.reshape(self.state[:,-1:], [-1, 1, 1, 1])                            #   x, y,  1
                # c2_v1 = tf.layers.conv2d(l2_in, filters=16, kernel_size=[1, 1], strides=[1, 1], activation=tf.nn.relu6)   #   x, y/2,  32
                pool1 = tf.layers.max_pooling2d(c1_v2, pool_size = [1, 2], strides=[1,1])                                   #   x, y/10, 32
                flat1 = tf.reshape(pool1, [-1, self.t_num  * 128])
                # flat2 = tf.reshape(c2_v1, [-1, 16])
                # l_con = tf.concat([flat1, flat2], 1)
                l1 = tf.layers.dense(flat1, self.t_num * 128, tf.nn.relu, name = "hidden_layer1")
                # l2 = tf.layers.dense(l1, self.a_dim * 8, tf.nn.relu, name = "hidden_layer2")
                out = tf.layers.dense(l1, 1,  name = "value")

                self.value = out
                self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Critic/Net")

            with tf.variable_scope("Loss"):
                self.td_error = tf.subtract(self.value_target, self.value)
                self.loss = tf.reduce_mean(tf.square(self.td_error))


            with tf.variable_scope('Train'):
                self.gradients = tf.gradients(self.loss, self.parameters)
                opt = tf.train.RMSPropOptimizer(self.l_r, name='RMSProp')
                self.update = opt.apply_gradients(zip(self.gradients, self.parameters))

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

