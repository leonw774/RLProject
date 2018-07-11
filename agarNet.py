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

INPUT_SHAPE = (scrshots_w, scrshots_h, 3)

game_region = (60, 100, screen_w - 120, screen_h - 200)
isdead_region = (screen_w - 200, screen_h - 70, 200, 30)

angle_devision = 24

def isdead() :
    isdead = pyautogui.screenshot(region = isdead_region)
    bbox = ImageChops.difference(dead_image, isdead).getbbox()
    return (bbox == None)
    
def get_screenshots(n, size) :
    squence_scrshot = np.zeros((n, size[0], size[1], 3))
    for i in range(n) :
        scrshot = (pyautogui.screenshot(region = game_region)).resize(size, resample = 0)
        #scrshot.save("test" + str(i) + ".png")
        scrshot = np.asarray(scrshot) / 255.5
        squence_scrshot[i, :, :, :] = scrshot
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

def agarNet(scrshot_size, action_num) :
    input = Input(scrshot_size)
    x = Conv2D(4, (3, 3), padding = "valid", activation = "relu", data_format = "channels_last")(input)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    x = Conv2D(8, (3, 3), padding = "valid", activation = "relu", data_format = "channels_last")(x)
    x = MaxPooling2D((2, 2), padding = "same")(x)
    flat = Flatten()(x)
    action = Dense(action_num)(flat)
    model = Model(input, action)
    model.summary()
    return model

AgarNet = agarNet(INPUT_SHAPE, angle_devision + 1)
AgarNet.compile(loss = "mse", optimizer = "sgd")
    
# Countdown
for i in range(5) :
    print(5 - i)
    time.sleep(1.0)

start_time = datetime.datetime.now()
while(True) :
    squence_srcshot = get_screenshots(scrshots_num, (scrshots_w, scrshots_h))
    action = AgarNet.predict(squence_srcshot)
    action = np.argmax(action)
    print(action)
    do_control(action)
    if (isdead()) :
        print("isdead")
        break
    if ((datetime.datetime.now() - start_time).total_seconds() > 10) : break
