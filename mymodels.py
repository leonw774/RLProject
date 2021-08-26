import numpy as np
from configure import Configuration as cfg
from gameagent import Action

from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D

'''
Deterministic Policy Gradient Algorithms
'''
class ActorCritic:
    def __init__ (self, load_weight_name, scrshot_size):    
        self.scrshot_size = scrshot_size

        self.q_optimizer = optimizers.adam(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)
        self.c_optimizer = optimizers.adam(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)

        self.initModels()
        self.critic.compile(loss = "mse", optimizer = self.c_optimizer)
        self.qNet.compile(loss = "mse", optimizer = self.q_optimizer)
        
        if load_weight_name :
            self.actor.load_weights("actor_" + load_weight_name)
            self.critic.load_weights("critic_" + load_weight_name)
        
    def initModels(self):
        input_scrshots = Input(self.scrshot_size) # screen shot image
        input_action_parameters = Input((4,))

        ### Make Actor
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
        self.actor = Model(input_scrshots, conc_action)
        self.actor.summary()

        ### Make Critic
        x = Conv2D(8, (4, 4), padding = "valid", activation = "relu")(input_scrshots)
        x = MaxPooling2D((3, 3), padding = "same")(x)
        x = Conv2D(16, (4, 4), padding = "valid", activation = "relu")(x)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(32, (3, 3), padding = "valid", activation = "relu")(x)
        x = Flatten()(x)
        conc = Concatenate(axis = -1)([x, input_action_parameters])
        score = Dense(1)(conc)
        self.critic = Model([input_scrshots, input_action_parameters], score)
        self.critic.summary()
        
        self.critic.trainable = False
        input_scrshots = Input(self.scrshot_size)
        action = self.actor(input_scrshots)
        pred_q = self.critic([input_scrshots, action])
        self.qNet = Model(input_scrshots, pred_q)
        self.qNet.summary()
    
    def decision(self, cur_shot, temperature = 1.0):
        return Action(np.squeeze(self.actor.predict(cur_shot)))
    
    def learn(self, trn_s, trn_a, trn_r):
        # make next predicted reward array and train input array at same time
        new_q = np.zeros(cfg.train_size)
        for j in range(cfg.train_size):
            pred_new_q = self.qNet.predict(np.expand_dims(trn_s[j+1], axis=0))
            new_q[j] = trn_r[j] + cfg.gamma * pred_new_q
        trn_s = trn_s[:cfg.train_size]
        trn_a = trn_a[:cfg.train_size]
        closs = self.critic.train_on_batch([trn_s, trn_a], new_q)
        qloss = self.qNet.train_on_batch(trn_s, new_q)
        return (qloss, closs)
        
    def save(self, save_weight_name):
        self.actor.save(save_weight_name)
        # self.critic.save("critic_" +save_weight_name)
        
 # end class ActorCritic