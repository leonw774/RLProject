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
DEAD_IMAGE = Image.open("dead_screenshot.png")
GAME_REGION = (60, 100, SCREEN_W - 120, SCREEN_H - 200)
ISDEAD_REGION = (SCREEN_W / 2 - 20, SCREEN_H / 3 - 20, 40, 40)
SCRSHOTS_NUM = 1
SCRSHOTS_W = 128
SCRSHOTS_H = 128
Q_INPUT_SHAPE = (SCRSHOTS_W, SCRSHOTS_H, SCRSHOTS_NUM)

# Q FUNCTION SETTING
EPOCHES = 1
LIMIT_TIME = 10 # in second

# REWARD SETTING
GAMMA = 1 / math.exp(1)

# ACTION SETTING
MOUSE_ANGLE_DEVISION = 16
MOUSE_ACTION_NUM = MOUSE_ANGLE_DEVISION + 1
#{angle_0, angle_1, ..., angle_n, no_angle}
KEYBOARD_ACTION_NUM = 0
#{key_0, key_1, ..., key_n}

def is_dead() :
    isdead = pyautogui.screenshot(region = ISDEAD_REGION)
    bbox = ImageChops.difference(DEAD_IMAGE, isdead).getbbox()
    return (bbox == None)
    
def get_screenshots() :
    squence_scrshot = np.zeros((1, SCRSHOTS_W, SCRSHOTS_H, SCRSHOTS_NUM))
    for i in range(SCRSHOTS_NUM) :
        scrshot = (pyautogui.screenshot(region = GAME_REGION)).convert('L').resize((SCRSHOTS_W, SCRSHOTS_H), resample = 0)
        #scrshot.save("test" + str(i) + ".png")
        scrshot = np.asarray(scrshot) / 255.5
        squence_scrshot[:, :, :, i] = scrshot
        time.sleep(0.1)
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
    input_action = Input((action_size,)) # one-hot
    x = Conv2D(4, (3, 3), padding = "valid", activation = "relu", data_format = "channels_last")(input_image)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    x = Conv2D(8, (3, 3), padding = "valid", activation = "relu", data_format = "channels_last")(x)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    flat_image = Flatten()(x)
    conc = Concatenate()([flat_image, input_action])
    score = Dense(1, activation = "relu")(conc)
    model = Model([input_image, input_action], score)
    model.summary()
    return model

# Q and Q_target
QNet = QNet(Q_INPUT_SHAPE, MOUSE_ACTION_NUM)
QNet.compile(loss = "mse", optimizer = "sgd")

# Q_target = QNet(Q_INPUT_SHAPE, MOUSE_ACTION_NUM)
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
for i in range(MOUSE_ACTION_NUM) :
    action_onehot = np.zeros((1, MOUSE_ACTION_NUM))
    action_onehot[0, i] = 1
    actionSet.append(action_onehot)

# States Queue
class StateQueue() :
    '''
    NOT DONE YET
    StateQueue is a list of step tuple: [ (scrshots, action, reward, next_scrshots), ... ]
    '''
    def __init__(self) :
        self.stepList = []
    
    def addStep(self, scrshots, action, reward, next_scrshots) :
        if (step.size == 3) :
            if (scrshots == (1, SCRSHOTS_W, SCRSHOTS_H, SCRSHOTS_NUM) # scrshots
                and action == (1, MOUSE_ACTION_NUM) # action
                and type(reward) == float # reward
                and next_scrshots.shape == (1, SCRSHOTS_W, SCRSHOTS_H, SCRSHOTS_NUM) # next_scrshots
               ):
                stepList.append((scrshots, action, reward, next_scrshots))
                return
        print("error in step tuple")
    
    def getStep(self, stepNum) :
    # step counted from zero.
        try :
            outStep = list(stepList[stepNum])
            return outStep
        except :
            print("stepNum out of size")

# Countdown
for i in range(3) :
    print(3 - i)
    time.sleep(1.0)

start_time = datetime.datetime.now()

for e in range(EPOCHES) :
    prev_squence_srcshot = get_screenshots()
    
    while(True) :
        
        cur_squence_srcshot = get_screenshots()
        max_pred_score = 0
        action = 0
        for i in range(MOUSE_ACTION_NUM) :
            pred_score = QNet.predict([cur_squence_srcshot, actionSet[i]])
            if (max_pred_score > pred_score) : action = i
        print(action)
        do_control(action)
        if (is_dead()) :
            print("isdead")
            break
        if ((datetime.datetime.now() - start_time).total_seconds() > LIMIT_TIME) : break
