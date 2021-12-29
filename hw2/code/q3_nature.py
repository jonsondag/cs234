import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 
        with tf.variable_scope(scope, reuse=reuse):
            # architecture from "Playing Atari with Deep RL"
            output1 = tf.layers.conv2d(inputs=state, filters=16, kernel_size=[8, 8], strides=[4, 4], padding='valid', activation=tf.nn.relu)
            output2 = tf.layers.conv2d(inputs=output1, filters=32, kernel_size=[4, 4], strides=[2, 2], padding='valid', activation=tf.nn.relu)
            flattened_output2 = tf.layers.flatten(output2)
            output3 = tf.layers.dense(inputs=flattened_output2, units=256, activation=tf.nn.relu)
            out = tf.layers.dense(inputs=output3, units=num_actions)
            # architecture from "Human-level control through deep RL"
            # weights1 = tf.get_variable('weights1', [8, 8, 4, 32], initializer=tf.random_normal_initializer())
            # biases1 = tf.get_variable('biases1', [32], initializer=tf.random_normal_initializer())
            # layer1 = tf.nn.conv2d(state, weights1, [1, 4, 4, 1], 'VALID')
            # output1 = tf.nn.relu(layer1 + biases1)
            # weights2 = tf.get_variable('weights2', [4, 4, 32, 64], initializer=tf.random_normal_initializer())
            # biases2 = tf.get_variable('biases2', [64], initializer=tf.random_normal_initializer())
            # layer2 = tf.nn.conv2d(output1, weights2, [1, 2, 2, 1], 'VALID')
            # output2 = tf.nn.relu(layer2 + biases2)
            # weights3 = tf.get_variable('weights3', [3, 3, 64, 64], initializer=tf.random_normal_initializer())
            # biases3 = tf.get_variable('biases3', [64], initializer=tf.random_normal_initializer())
            # layer3 = tf.nn.conv2d(output2, weights3, [1, 1, 1, 1], 'VALID')
            # output3 = tf.nn.relu(layer3 + biases3)
            # flattened_output3 = tf.layers.flatten(output3)
            # layer4 = tf.layers.dense(flattened_output3, 512, tf.nn.relu)
            # out = tf.layers.dense(layer4, num_actions)
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
