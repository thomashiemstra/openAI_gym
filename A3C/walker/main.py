import multiprocessing as mp
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from matplotlib import animation
import pickle
import datetime

GAME = 'Pendulum-v0'
GLOBAL_STEPS = 3000
UPDATE_GLOBAL_ITER = 30
GAMMA = 0.9
ENTROPY_BETA = 0.01
LEARNING_RATE = 0.0001    # learning rate for actor
END_LEARNING_RATE = 0.0001
EPSILON = 1e-5
GRAD_CLIP = 0.1
DECAY = 0.99

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):
    sess = None

    def __init__(self, scope, opt_a=None, opt_c=None, global_net=None, global_step=None):
        if scope == 'global_net':  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.global_step = global_step
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu, sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = opt_a.apply_gradients(zip(self.a_grads, global_net.a_params))
                    self.update_c_op = opt_c.apply_gradients(zip(self.c_grads, global_net.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')*A_BOUND[1]
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})


def work(job_name, task_index, global_ep, r_local_queue, global_running_r, local_latch):
    chief = (task_index == 0)

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
        env = gym.make(GAME).unwrapped
        print('Start Worker: ', task_index)
        with tf.device(tf.train.replica_device_setter(ps_device="/job:ps/cpu:0",
                worker_device="/job:worker/task:%d/cpu:0" % task_index,
                cluster=cluster)):

            global_step = tf.train.get_or_create_global_step(graph=None)
            adaptive_learning_rate = tf.train.polynomial_decay(LEARNING_RATE, global_step, GLOBAL_STEPS, END_LEARNING_RATE)

            opt_a = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=DECAY, name='opt_a')
            opt_c = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=DECAY, name='opt_a')

            global_net = ACNet('global_net')

            local_net = ACNet('local_ac%d' % task_index, opt_a, opt_c, global_net, global_step)

        # terrible hack
        checkpoint_dir = None
        # if chief:
        #     checkpoint_dir = "C:/Users/thomas/PycharmProjects/openAI_gym/A3C/walker/tmp"

        # set training steps
        hooks = [tf.train.StopAtStepHook(last_step=100000)]
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               checkpoint_dir=checkpoint_dir,
                                               is_chief=True,
                                               log_step_count_steps=0,
                                               hooks=hooks,
                                               config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
            print('Start Worker Session: ', task_index)
            local_net.sess = sess
            # terrible hack 2.0
            # if chief:
            #     local_net.push_to_global()
            # else:
            #     time.sleep(2)
            #     local_net.pull_global()

            buffer_s, buffer_a, buffer_r = [], [], []

            while global_ep.value < GLOBAL_STEPS:
                s = env.reset()
                ep_r = 0
                done = False
                total_step = 0
                while not done:
                    a = local_net.choose_action(s)
                    s_, reward, done, info = env.step(a)
                    done = True if total_step == 200 - 1 else False

                    ep_r += reward
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append((reward + 8) / 8)

                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                        if done:
                            v_s_ = 0  # terminal
                        else:
                            v_s_ = sess.run(local_net.v, {local_net.s: s_[np.newaxis, :]})[0, 0]
                        buffer_v_target = []
                        for r in buffer_r[::-1]:  # reverse buffer r
                            v_s_ = r + GAMMA * v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()

                        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                            buffer_v_target)
                        feed_dict = {
                            local_net.s: buffer_s,
                            local_net.a_his: buffer_a,
                            local_net.v_target: buffer_v_target,
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
                            # local_net.increment_global_step()

                        print(
                            "Task: %i" % task_index,
                            "| Ep: %i" % global_ep.value,
                            "| Ep_r: %i" % ep_r,
                            "| global_r %i" % global_running_r.value,
                            "| total step %i" % total_step
                        )



            print('Worker Done: ', task_index, time.time()-t1)

            if task_index == 0:
                frames = []
                env = gym.make(GAME)

                s = env.reset()
                done = False
                while not done:
                    frames.append(env.render(mode='rgb_array'))
                    a = local_net.choose_action(s)
                    print(a)
                    s, reward, done, info = env.step(a)

                env.close()
                save_frames_as_gif(frames)
                with open('test.pkl', 'wb') as output:
                    pickle.dump(frames, output, pickle.HIGHEST_PROTOCOL)

            with local_latch.get_lock():
                local_latch.value -= 1


def save_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    now = datetime.datetime.now()

    file_name = 'output_' + now.strftime("%m-%d_%H:%M:%S") + '.html'
    anim.save(file_name, writer='html', fps=60)


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
