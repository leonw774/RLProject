import numpy as np
from configure import Configuration as cfg
from gameagent import Action
from stepqueue import StepQueue

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau = 0.005):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

# for exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

'''
Deterministic Policy Gradient Algorithms
http://proceedings.mlr.press/v32/silver14.pdf
'''
class ActorCritic:
    def __init__ (self, load_weight_name, scrshot_size):
        self.scrshot_size = scrshot_size
        self.step_queue = StepQueue()

        self.c_optimizer = optimizers.adam(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)
        self.a_optimizer = optimizers.adam(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)

        self.init_models()
        self.actor.compile(optimizer = self.a_optimizer)
        self.valueNet.compile(loss = "mse", optimizer = self.c_optimizer)
        
        if load_weight_name :
            self.actor.load_weights("actor_" + load_weight_name)
            self.critic.load_weights("critic_" + load_weight_name)
        
    def init_models(self):
        input_scrshots = Input(self.scrshot_size) # screen shot image
        input_action_parameters = Input((4,))

        self.actor = self.make_actor()
        self.critic = self.make_critic()

        input_scrshots = Input(self.scrshot_size)
        a = self.actor(input_scrshots)
        q = self.critic([input_scrshots, a])
        self.valueNet = Model(input_scrshots, q)

        target_actor = self.make_actor()
        target_critic = self.make_actor()

        t_input_scrshots = Input(self.scrshot_size)
        t_a = target_actor(t_input_scrshots)
        t_q = target_critic([t_input_scrshots, t_a])
        self.target_valueNet = Model(t_input_scrshots, t_q)
        
        self.target_valueNet.set_weights(self.valueNet.get_weights())

    def make_actor(self):
        input_scrshots = Input(self.scrshot_size) # screen shot image
        # Actor is P(s)
        a = Conv2D(8, (4, 4), padding = "valid", activation = "relu")(input_scrshots)
        a = MaxPooling2D((3, 3), padding = "same")(a)
        a = Conv2D(16, (4, 4), padding = "valid", activation = "relu")(a)
        a = MaxPooling2D((2, 2), padding = "same")(a)
        a = Conv2D(32, (3, 3), padding = "valid", activation = "relu")(a)
        a = Flatten()(a)
        time = Dense(1, activation = "relu")(a)
        speed = Dense(1, activation = "relu")(a)
        initial_angle = Dense(1, activation = "sigmoid")(a)
        vertical_acc = Dense(1, activation = "tanh")(a)
        conc_action = Concatenate(axis = -1)([time, speed, initial_angle, vertical_acc])
        actor = Model(input_scrshots, conc_action)
        return actor

    def make_critic(self):
        # Critic is Q(s, a)
        input_scrshots = Input(self.scrshot_size) # screen shot image
        input_action_parameters = Input((4,)) # action representation
        x = Conv2D(8, (4, 4), padding = "valid", activation = "relu")(input_scrshots)
        x = MaxPooling2D((3, 3), padding = "same")(x)
        x = Conv2D(16, (4, 4), padding = "valid", activation = "relu")(x)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(32, (3, 3), padding = "valid", activation = "relu")(x)
        x = Flatten()(x)
        conc = Concatenate(axis = -1)([x, input_action_parameters])
        score = Dense(1)(conc)
        critic = Model([input_scrshots, input_action_parameters], score)
        return critic

    
    def push_step(self, scrshot, action, reward):
        self.step_queue.push_step(scrshot, action, reward)
        if len(self.step_queue) > cfg.stepqueue_length_max:
            self.step_queue = self.step_queue[1:]

    def decision(self, cur_shot, temperature = 1.0):
        a = np.squeeze(self.actor.predict(cur_shot))
        noise = OUActionNoise()(np.zeros(4), 0.2)
        return Action()
    
    def learn(self):
        # select a random sequence of length of (train_size + 1)
        random_id = np.random.randint(len(self.step_queue) - cfg.train_size)
        trn_s, trn_a, trn_r = self.step_queue.get_steps(random_id, cfg.train_size + 1)
        new_v = np.zeros(cfg.train_size)
        next_v = self.target_valueNet.predict(np.expand_dims(trn_s[:-1], axis=0))
        new_v = trn_r[:-1] + cfg.gamma * next_v
        trn_s = trn_s[:-1]

        with tf.GradientTape() as tape:
            v = self.valueNet(np.expand_dims(trn_s, axis=0))

            # train critic
            c_loss = tf.math.reduce_mean(tf.math.square(new_v - v)) # TDError
            c_grad = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_grad, self.valueNet.trainable_variables))

            # train actor
            a_loss = -tf.math.reduce_mean(v) # maxmize v
            a_grad = tape.gradient(a_loss, self.actor.trainable_variables)
            self.a_optimizer.apply_gradients(zip(a_grad, self.actor.trainable_variables))

        # update target network
        update_target(self.target_valueNet.variables, self.valueNet.variables)

        return (c_loss, a_loss)
        
    def save(self, save_weight_name):
        self.actor.save(save_weight_name)
        # self.critic.save("critic_" +save_weight_name)
        
 # end class ActorCritic
