"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.
The Cartpole example using distributed tensorflow + multiprocessing.
View more on my tutorial page: https://morvanzhou.github.io/
"""

import multiprocessing as mp
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import tensorflow as tf

GLOBAL_STEPS = 1000
UPDATE_GLOBAL_ITER = 30
GAMMA = 0.9
ENTROPY_BETA = 0.01
LEARNING_RATE = 0.001    # learning rate for actor
EPSILON = 1e-5
GRAD_CLIP = 1
DECAY = 0.99

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet(object):
    sess = None

    def __init__(self, scope, opt_a=None, global_net=None):
        if scope == 'global_net':  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                _, _, self.params = self._build_net(scope)
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.discounted_reward = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.value, self.params = self._build_net(scope)

                advantage = tf.subtract(self.discounted_reward, self.value, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_sum(tf.square(advantage))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(
                        tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32))
                    policy_loss = - tf.reduce_sum(log_prob * tf.stop_gradient(advantage))
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5))  # encourage exploration

                    self.a_loss = policy_loss - ENTROPY_BETA * entropy

                self.loss = self.a_loss + 0.5 * self.c_loss

                with tf.name_scope('local_grad'):
                    self.grads_unclipped = tf.gradients(self.loss, self.params)
                    self.grads, _ = tf.clip_by_global_norm(self.grads_unclipped, 0.1)

            self.global_step = tf.train.get_or_create_global_step()
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, global_net.params)]

                with tf.name_scope('push'):
                    self.update_a_op = opt_a.apply_gradients(zip(self.grads, global_net.params), global_step=self.global_step)

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('model'):

            l_s_1 = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='la')

            a_prob = tf.layers.dense(l_s_1, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            value = tf.layers.dense(l_s_1, 1, kernel_initializer=w_init, name='v')  # state value

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/model')
        return a_prob, value, params

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def update_global(self, feed_dict):  # run by a local
        self.sess.run(self.update_a_op, feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run(self.pull_a_params_op)


def work(job_name, task_index, global_ep, r_local_queue, global_running_r, local_latch):
    import tensorflow as tf
    # set work's ip:port
    cluster = tf.train.ClusterSpec({
        "ps": ['localhost:2220'],
        "worker": ['localhost:2221', 'localhost:2222', 'localhost:2223',
                   'localhost:2224', 'localhost:2225', 'localhost:2226',
                   'localhost:2227', 'localhost:2228', 'localhost:2229',
                   # 'localhost:2230', 'localhost:2231', 'localhost:2232',
                   ]
    })

    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
    if job_name == 'ps':
        print('Start Parameter Sever: ', task_index)
        server.join()
    else:
        t1 = time.time()
        env = gym.make('CartPole-v0').unwrapped
        print('Start Worker: ', task_index)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            opt_a = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=DECAY, name='opt_a')
            global_net = ACNet('global_net')

        local_net = ACNet('local_ac%d' % task_index, opt_a, global_net)

        # set training steps
        hooks = [tf.train.StopAtStepHook(last_step=100000)]
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=True,
                                               hooks=hooks,) as sess:
            print('Start Worker Session: ', task_index)
            local_net.sess = sess
            buffer_s, buffer_a, buffer_r = [], [], []

            while global_ep.value < GLOBAL_STEPS:
                s = env.reset()
                ep_r = 0
                done = False
                total_step = 0
                while not done:
                    a = local_net.choose_action(s)
                    s_, reward, done, info = env.step(a)

                    if done:
                        reward = -1
                    if total_step > 400:
                        reward = 10
                        done = True

                    ep_r += reward
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(reward)

                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                        if done:
                            v_s_ = 0  # terminal
                        else:
                            v_s_ = sess.run(local_net.value, {local_net.s: s_[np.newaxis, :]})[0, 0]
                        buffer_v_target = []
                        buffer_r.reverse()
                        for r in buffer_r:  # reverse buffer r
                            v_s_ = r + GAMMA * v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()

                        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                            buffer_v_target)
                        feed_dict = {
                            local_net.s: buffer_s,
                            local_net.a_his: buffer_a,
                            local_net.discounted_reward: buffer_v_target,
                        }
                        local_net.update_global(feed_dict)
                        buffer_s, buffer_a, buffer_r = [], [], []
                        local_net.pull_global()
                    s = s_
                    total_step += 1
                    if done:
                        with global_running_r.get_lock():
                            global_running_r.value = .99 * global_running_r.value + 0.01 * ep_r
                            try:
                                r_local_queue.put(global_running_r.value, True, 1)
                            except:
                                print("Queue is fucked")

                        with global_ep.get_lock():
                            global_ep.value += 1

                        print(
                            "Task: %i" % task_index,
                            "| Ep: %i" % global_ep.value,
                            "| Ep_r: %i" % ep_r,
                            "| global_r %i" % global_running_r.value,
                            "| total step %i" % total_step,
                        )

            print('Worker Done: ', task_index, time.time()-t1)

            if task_index == 0:
                env = gym.make('CartPole-v0')

                s = env.reset()
                done = False
                while not done:
                    a = local_net.choose_action(s)
                    s, reward, done, info = env.step(a)
                    env.render()

                env.close()

            with local_latch.get_lock():
                local_latch.value -= 1


def queue_flusher(q, latch):
    res = []
    while True:
        with latch.get_lock():
            running = latch.value != 0
        while not q.empty():
            res.append(q.get(True, 1))
        time.sleep(0.5)
        if not running:
            break
    return res


def get_num_workers(jobs):
    res = 0
    for name, i in jobs:
        if name == 'worker':
            res += 1
    return res


if __name__ == "__main__":
    # use multiprocessing to create a local cluster with 2 parameter servers and 4 workers
    global_ep = mp.Value('i', 0)
    r_queue = mp.Queue()
    global_running_r = mp.Value('d', 0)

    jobs = [
        ('ps', 0),
        ('worker', 0), ('worker', 1), ('worker', 2),
        ('worker', 3), ('worker', 4), ('worker', 5),
        ('worker', 6), ('worker', 7), ('worker', 8),
        # ('worker', 9), ('worker', 10), ('worker', 11),
    ]

    num_workers = get_num_workers(jobs)

    latch = mp.Value('i', num_workers)

    processes = [mp.Process(target=work, args=(j, i, global_ep, r_queue, global_running_r, latch), name="task:%d" % i) for j, i in jobs]
    [p.start() for p in processes]

    ep_r = queue_flusher(r_queue, latch)

    print("queue emptied")

    print("everyone has joined")

    plt.plot(np.arange(len(ep_r)), ep_r)
    plt.title('Distributed training')
    plt.xlabel('Step')
    plt.ylabel('Total moving reward')
    plt.show()

    print("KILL!")
    [p.terminate() for p in processes]
