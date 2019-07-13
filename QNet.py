import numpy as np
from configure import Configuration as cfg
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, SimpleRNN, Input, MaxPooling2D, MaxPooling3D, LeakyReLU

# Q-network: guess how many score a image will make
class QNet() :
    def __init__ (self, use_target, load_weight_name, scrshot_size, action_size) :    
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
    
    def decision(self, cur_shot, tempature = 0.5)
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
        
        