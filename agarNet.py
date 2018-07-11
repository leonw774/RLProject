import numpy as np
from PIL import ImageChops, Image
import pyautogui
import time
import math
import random
import datetime

from keras import optimizers
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Concatenate, Conv2D, Conv3D, Dense, Dropout, Flatten, SimpleRNN, Input, MaxPooling2D, MaxPooling3D

screen_w, screen_h = pyautogui.size()
dead_image = Image.open("dead_screenshot.png")

scrshots_num = 2
scrshots_w = 128
scrshots_h = 128

net_input_shape = (scrshots_w, scrshots_h, 3 * scrshots_num)

game_region = (60, 100, screen_w - 120, screen_h - 200)
isdead_region = (screen_w / 2 - 20, screen_h / 3 - 20, 40, 40)

angle_devision = 24
action_num = angle_devision + 1

def isdead() :
    isdead = pyautogui.screenshot(region = isdead_region)
    bbox = ImageChops.difference(dead_image, isdead).getbbox()
    return (bbox == None)
    
def get_screenshots(n, size) :
    squence_scrshot = np.zeros((1, size[0], size[1], 3 * n))
    for i in range(n) :
        scrshot = (pyautogui.screenshot(region = game_region)).resize(size, resample = 0)
        #scrshot.save("test" + str(i) + ".png")
        scrshot = np.asarray(scrshot) / 255.5
        squence_scrshot[:, :, :, 3 * i : 3 * (i+1)] = scrshot
        time.sleep(0.1)
    return squence_scrshot
    
def do_control(control_id) : 
    if (control_id >= angle_devision) :
        pyautogui.moveTo(screen_w / 2, screen_h / 2)
    else :
        angle = 2 * math.pi * control_id / angle_devision
        offset_x = math.cos(angle) * 100
        offset_y = math.sin(angle) * 100
        pyautogui.moveTo(offset_x + screen_w / 2, offset_y + screen_h / 2)
    return

# guess how many score it will make
def agarNet(scrshot_size, action_size) :
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

AgarNet = agarNet(net_input_shape, action_num)
AgarNet.compile(loss = "mse", optimizer = "sgd")
    
# Countdown
for i in range(3) :
    print(3 - i)
    time.sleep(1.0)

ActionSet = []
for i in range(action_num) :
    action_onehot = np.zeros((1, action_num))
    action_onehot[0, i] = 1
    ActionSet.append(action_onehot)
print(ActionSet[3].shape)

StateQueue = []

start_time = datetime.datetime.now()
while(True) :
    squence_srcshot = get_screenshots(scrshots_num, (scrshots_w, scrshots_h))
    max_pred_score = 0
    action = 0
    for i in range(action_num) :
        pred_score = AgarNet.predict([squence_srcshot, ActionSet[i]])
        if (max_pred_score > pred_score) : action = i
    print(action)
    do_control(action)
    if (isdead()) :
        print("isdead")
        break
    if ((datetime.datetime.now() - start_time).total_seconds() > 30) : break
