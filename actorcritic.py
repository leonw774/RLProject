from configure import Configuration as cfg
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, Input, LeakyReLU

class ActorCritic:
    def __init__ (self, a_load_weight_name, c_load_weight_name, scrshot_size, action_size) :    
        
        self.actor = self.make_actor(a_load_weight_name)
        self.actor_optimizer = optimizers.rmsprop(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)
        self.actor.compile(loss = self.actor_loss, optimizer = self.model_optimizer, metrics = ['mse'])
        
        self.critic = self.make_actor(a_load_weight_name)
        self.critic_optimizer = optimizers.adam(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)
        self.critic.compile(loss = "mse", optimizer = self.model_optimizer, metrics = ['mse'])
        
        if a_load_weight_name :
            self.actor.load_weights(a_load_weight_name)
        if c_load_weight_name :
            self.critic.load_weights(c_load_weight_name)
        
    def make_actor(self, load_weight_name) :
        input_scrshots = Input(scrshot_size) # screen shot image
        x = Conv2D(16, (4, 4), padding = "valid", activation = "relu")(input_scrshots)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(32, (4, 4), padding = "valid", activation = "relu")(x)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(64, (3, 3), padding = "valid", activation = "relu")(x)
        x = Flatten()(x)
        x = Dense(128, activation = "relu")(x)
        probs = Dense(action_size)(x)
        model = Model(input_scrshots, probs)
        model.summary()
        return model
    
    def make_critic(self, load_weight_name) :
        input_scrshots = Input(scrshot_size) # screen shot image
        x = Conv2D(16, (4, 4), padding = "valid", activation = "relu")(input_scrshots)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(32, (4, 4), padding = "valid", activation = "relu")(x)
        x = MaxPooling2D((2, 2), padding = "same")(x)
        x = Conv2D(64, (3, 3), padding = "valid", activation = "relu")(x)
        x = Flatten()(x)
        x = Dense(128, activation = "softmax")(x)
        scores = Dense(action_size)(x)
        model = Model(input_scrshots, scores)
        model.summary()
        return model
        
    def decision(self, cur_shot, tempature = 1.0)
        pred = np.squeeze(self.actor.predict(cur_shot))
        pred = np.log(prediction) / temperature
        exp_preds = np.exp(prediction)
        pred = exp_preds / np.sum(exp_preds)
        return np.random.multinomial(1, pred, 1)
    
    def actor_loss(y_true, y_pred) :
    '''
        y_true: (train_action, td_error)
        y_pred: action_prob
    '''
        trn_a, td_error = y_true[:, 0], y_true[:, 1]
        loss = K.binary_crossentropy(trn_a, y_pred) * td_error
        return loss
    
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
        
        closs = self.critic.train_on_batch(trn_s[:cfg.train_size], new_r)
        
        td_error = new_r - trn_r
        actor_y_true = np.array((trn_a, td_error))
        aloss = self.actor.train_on_batch(trn_s[:cfg.train_size], actor_y_true)
        return aloss, closs
        
    def save(self, a_save_weight_name, c_save_weight_name) :
        self.actor.save(a_save_weight_name)
        self.critic.save(c_save_weight_name)
        
        
    