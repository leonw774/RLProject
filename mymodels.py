import numpy as np
from configure import Configuration as cfg

from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D

# Q-network: guess how many score a image will make
class QNet() :
    def __init__ (self, load_weight_name, scrshot_size, action_size, use_target = cfg.use_target_model) :    
        input_scrshots = Input(scrshot_size) # screen shot image
        x = Conv2D(32, (6, 6), padding = "valid", activation = "relu")(input_scrshots)
        x = MaxPooling2D((4, 4), padding = "same")(x)
        x = Conv2D(64, (4, 4), padding = "valid", activation = "relu")(x)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(128, (3, 3), padding = "valid", activation = "relu")(x)
        x = Flatten()(x)
        x = Dense(128, activation = "relu")(x)
        scores = Dense(action_size)(x)
        
        self.model = Model(input_scrshots, scores)
        self.model.summary()
        self.model_optimizer = optimizers.rmsprop(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)
        self.model.compile(loss = "mse", optimizer = self.model_optimizer, metrics = ['mse'])
        
        if load_weight_name :
            self.model.load_weights(load_weight_name)
        
        if use_target :
            self.model_target.set_weights(self.model.get_weight())
        else :
            self.model_target = None
    
    def decision(self, cur_shot, tempature = 0.5) :
        predQ = np.squeeze(self.model.predict(cur_shot))
        # soft max
        w = predQ / tempature
        w = np.exp(w)
        w /= w.sum()
        return np.random.choice(np.arange(cfg.actions_num), p = w)
    
    def learn(self, trn_s, trn_a, trn_r) :
        # make next predicted reward array and train input array at same time
        new_r = np.zeros((cfg.train_size, cfg.actions_num))
        for j in range(cfg.train_size) :
            if self.model_target :
                new_r[j] = self.model_target.predict(np.expand_dims(trn_s[j], axis = 0))
                predict_Q = self.model_target.predict(np.expand_dims(trn_s[j+1], axis = 0))
            else :
                new_r[j] = self.model.predict(np.expand_dims(trn_s[j], axis = 0))
                predict_Q = self.model.predict(np.expand_dims(trn_s[j+1], axis = 0))
            new_r[j, trn_a[j]] = trn_r[j] + np.max(predict_Q) * cfg.gamma
            # Q_new = r + Q_predict(a,s) * gamma
        
        #print("new_r\n", new_r[0])
        return self.model.train_on_batch(trn_s[:cfg.train_size], new_r)[0]
        
    def save(self, save_weight_name):
        if self.model_target :
            self.model_target.set_weights(self.model.get_weights())
            self.model_target.save(save_weight_name)
        else :
            self.model.save(save_weight_name)
        
 # end class QNet
 
class ActorCritic :
    def __init__ (self, load_weight_name, scrshot_size, action_size) :    
        self.td_error = np.zeros((cfg.train_size))
        
        self.actor = self.init_actor(scrshot_size, action_size)
        self.actor_optimizer = optimizers.rmsprop(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)
        self.actor.compile(loss = self.actor_loss, optimizer = self.actor_optimizer)
        
        self.critic = self.init_critic(scrshot_size, action_size)
        self.critic_optimizer = optimizers.adam(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)
        self.critic.compile(loss = "mse", optimizer = self.critic_optimizer)
        
        if load_weight_name :
            self.actor.load_weights("actor_" + load_weight_name)
            self.critic.load_weights("critic_" + load_weight_name)
        
    def init_actor(self, scrshot_size, action_size) :
        input_scrshots = Input(scrshot_size) # screen shot image
        x = Conv2D(16, (4, 4), padding = "valid", activation = "relu")(input_scrshots)
        x = MaxPooling2D((3, 3), padding = "same")(x)
        x = Conv2D(32, (4, 4), padding = "valid", activation = "relu")(x)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(64, (3, 3), padding = "valid", activation = "relu")(x)
        x = Flatten()(x)
        probs = Dense(action_size, activation = "softmax")(x)
        model = Model(input_scrshots, probs)
        model.summary()
        return model
    
    def init_critic(self, scrshot_size, action_size) :
        input_scrshots = Input(scrshot_size) # screen shot image
        x = Conv2D(16, (4, 4), padding = "valid", activation = "relu")(input_scrshots)
        x = MaxPooling2D((3, 3), padding = "same")(x)
        x = Conv2D(32, (4, 4), padding = "valid", activation = "relu")(x)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(64, (3, 3), padding = "valid", activation = "relu")(x)
        x = Flatten()(x)
        scores = Dense(action_size)(x)
        model = Model(input_scrshots, scores)
        model.summary()
        return model
        
    def decision(self, cur_shot, temperature = 1.0) :
        pred = np.squeeze(self.actor.predict(cur_shot))
        pred = np.log(pred) / temperature
        exp_preds = np.exp(pred)
        pred = exp_preds / np.sum(exp_preds)
        return np.argmax(np.random.multinomial(1, pred, None))
    
    def actor_loss(self, y_true, y_pred) :
        '''
        y_true: train_action
        y_pred: action_prob
        '''
        #print(y_true.shape, y_pred.shape)
        loss = K.sparse_categorical_crossentropy(y_true, y_pred) # * self.td_error
        return loss
    
    def learn(self, trn_s, trn_a, trn_r) :
        # make next predicted reward array and train input array at same time
        new_r = np.zeros((cfg.train_size, cfg.actions_num))
        new_a = np.zeros((cfg.train_size))
        for j in range(cfg.train_size) :
            predict_cur = self.critic.predict(np.expand_dims(trn_s[j], axis = 0))
            predict_next = self.critic.predict(np.expand_dims(trn_s[j+1], axis = 0))
            new_r[j] = predict_cur
            new_r[j, trn_a[j]] = trn_r[j] + np.max(predict_next) * cfg.gamma
            new_a[j] = np.argmax(predict_cur)
            self.td_error[j] = new_r[j, trn_a[j]] - np.max(predict_cur)
        #print(self.td_error)
        closs = self.critic.train_on_batch(trn_s[:cfg.train_size], new_r)
        aloss = self.actor.train_on_batch(trn_s[:cfg.train_size], new_a)
        #print("a", aloss)
        return (aloss, closs)
        
    def save(self, save_weight_name) :
        self.actor.save("actor_" + save_weight_name)
        self.critic.save("critic_" +save_weight_name)
        
 # end class ActorCritic