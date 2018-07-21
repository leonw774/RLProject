import numpy as np
import pyautogui
import time
import math
import random
import datetime

from PIL import ImageChops, Image
from keras import optimizers
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, SimpleRNN, Input, MaxPooling2D, MaxPooling3D

# SCREENSHOTS SETTING
SCREEN_W, SCREEN_H = pyautogui.size()
GAME_REGION = (60, 100, SCREEN_W - 120, SCREEN_H - 200)
ISDEAD_REGION = (SCREEN_W / 2 - 20, SCREEN_H / 3 - 20, 40, 40)
DEAD_IMAGE = Image.open("isdead_image.png")
SCRSHOTS_NUM = 1
SCRSHOTS_W = 96
SCRSHOTS_H = 54
Q_INPUT_SHAPE = (SCRSHOTS_H, SCRSHOTS_W, SCRSHOTS_NUM)

# Q FUNCTION SETTING
EPOCHES = 2

# REWARD SETTING
GAMMA = 1 / math.exp(1)

# ACTION SETTING
EPSILON = 0.3
MOUSE_ANGLE_DEVISION = 16
MOUSE_ACTION_NUM = MOUSE_ANGLE_DEVISION + 1 #{angle_0, angle_1, ..., angle_n, no_angle}
KEYBOARD_ACTION_NUM = 0 #{key_0, key_1, ..., key_n}
TOTAL_ACTION_NUM = MOUSE_ACTION_NUM + KEYBOARD_ACTION_NUM

# TRAINING SETTING
BATCH_NUM = 8
    
def get_screenshots() :
    squence_scrshot = np.zeros((1, SCRSHOTS_H, SCRSHOTS_W, SCRSHOTS_NUM))
    for i in range(SCRSHOTS_NUM) :
        scrshot = (pyautogui.screenshot(region = GAME_REGION)).convert('L').resize((SCRSHOTS_W, SCRSHOTS_H), resample = 0)
        #scrshot.save("test" + str(i) + ".png")
        scrshot = np.asarray(scrshot) / 255.5
        squence_scrshot[0, :, :, i] = scrshot
        time.sleep(0.01)
    return squence_scrshot
    
def do_control(control_id) : 
    if (control_id >= MOUSE_ANGLE_DEVISION) :
        pyautogui.moveTo(SCREEN_W / 2, SCREEN_H / 2)
    else :
        angle = 2 * math.pi * control_id / MOUSE_ANGLE_DEVISION
        offset_x = math.cos(angle) * 100
        offset_y = math.sin(angle) * 100
        pyautogui.moveTo(offset_x + SCREEN_W / 2, offset_y + SCREEN_H / 2)
    return

# Q-network: guess how many score it will make
def QNet(scrshot_size, action_size) :
    input_image = Input(scrshot_size) # image
    input_action = Input((TOTAL_ACTION_NUM,)) # one-hot
    x = Conv2D(4, (3, 3), padding = "valid", activation = "relu", data_format = "channels_last")(input_image)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    x = Conv2D(8, (3, 3), padding = "valid", activation = "relu", data_format = "channels_last")(x)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    flat_image = Flatten()(x)
    #flat_action = Flatten()(input_action)
    conc = Concatenate()([flat_image, input_action])
    score = Dense(1, activation = "relu")(conc)
    model = Model([input_image, input_action], score)
    model.summary()
    return model

# Q and Q_target
QNet = QNet(Q_INPUT_SHAPE, TOTAL_ACTION_NUM)
QNet.compile(loss = "mse", optimizer = "sgd")

# Q_target = QNet(Q_INPUT_SHAPE, TOTAL_ACTION_NUM)
# Q_target.set_weights(model.get_weights())

'''
We will train Q, at time t, as:
    y_pred = Q([state_t, a_t])
    y_true = r_t + gamma * max(Q_target([state_t, for a in A]))
update Q's weight in mse
after a number of steps, copy Q to Q_target.
'''

# Action Set
actionSet = []
for i in range(TOTAL_ACTION_NUM) :
    action_onehot = np.zeros((1, TOTAL_ACTION_NUM))
    action_onehot[0, i] = 1
    actionSet.append(action_onehot) 

# States Queue
class StateQueue() :
    '''
    NOT DONE YET
    StateQueue is a list of step tuple: [ (scrshots, action, reward, next_scrshots), ... ]
    '''
    def __init__(self) :
        self.scrshotList = []
        self.actionList = []
        self.rewardList = []
        self.nxtScrchotList = []
    
    def addStep(self, scrshots, action, reward, next_scrshots) :
        if (scrshots.shape == (1, SCRSHOTS_H, SCRSHOTS_W, SCRSHOTS_NUM) # scrshots
            and action.shape == (1, TOTAL_ACTION_NUM) # actionSet
            and next_scrshots.shape == (1, SCRSHOTS_H, SCRSHOTS_W, SCRSHOTS_NUM) # next_scrshots
           ):
            self.scrshotList.append(scrshots[0])
            self.actionList.append(action[0])
            self.rewardList.append(reward)
            self.nxtScrchotList.append(next_scrshots[0])
            return
        print("error in step tuple:")
        print(scrshots.shape, action, reward, next_scrshots.shape)
    
    def getStepAt(self, i) :
    # step counted from zero.
        try :
            return self.scrshotList[i], self.actionList[i], self.rewardList[i], self.nxtScrchotList[i]
        except :
            print("stepNum out of size")
            
    def getAllStepsInArray(self) :
        return np.array(self.scrshotList), np.array(self.actionList), np.array(self.rewardList), np.array(self.nxtScrchotList)

def is_dead() :
    isdead = pyautogui.screenshot(region = ISDEAD_REGION)
    isdead.save("last_isdead.png")
    bbox = ImageChops.difference(DEAD_IMAGE, isdead).getbbox()
    return (bbox == None)

# Start Countdown
countdown = 10
for i in range(countdown) :
    print(countdown - i)
    time.sleep(1.0)
start_time = datetime.datetime.now()

for e in range(EPOCHES) :
    stateQueue = StateQueue()
    
    for s in range(BATCH_NUM) :
        cur_scrshots = get_screenshots()
        if random.random() < EPSILON :
            cur_action = random.randrange(TOTAL_ACTION_NUM)
        else :
            q_values = [QNet.predict([cur_scrshots, actionSet[i]]) for i in range(TOTAL_ACTION_NUM)]
            cur_action = np.argmax(q_values)
        
        print(cur_action)
        do_control(cur_action)
        
        next_scrshots = get_screenshots()
        
        if (is_dead()) :
            cur_reward = -1.0
            print("isdead")
            break
        else :
            cur_reward = 1.0
        
        stateQueue.addStep(cur_scrshots, actionSet[cur_action], cur_reward, next_scrshots)
    # end for BATCH
    
    # Experience Replay
    train_inputs_scrshots, train_inputs_action, reward, next_scrshots = stateQueue.getAllStepsInArray()
    next_reward = np.zeros((BATCH_NUM, 1))
    for i in range(BATCH_NUM) :
        next_reward[i] = max([QNet.predict([np.reshape(next_scrshots[i], (1, SCRSHOTS_H, SCRSHOTS_W, SCRSHOTS_NUM)), actionSet[j]]) for j in range(TOTAL_ACTION_NUM)])
    
    train_targets = np.add(np.reshape(reward, (BATCH_NUM, 1)), np.multiply(GAMMA, next_reward))
    print(reward.shape, next_reward.shape, train_targets.shape)
    train_targets[-1] = reward[-1] # because isdead
    loss = QNet.train_on_batch([train_inputs_scrshots, train_inputs_action], train_targets)
    
    del stateQueue
    
# end for EPOCHES
