import numpy as np
import math
from pyautogui import screenshot
from PIL import Image

from time import sleep
from win32 import win32gui
from keras import optimizers
from directinputs import Keys
directInput = Keys()

def get_screen_rect(title = None):
    if title :
        gamewin = win32gui.FindWindow(None, title)
        if not gamewin:
            raise Exception('window title not found')
        #get the bounding box of the window
        x1, y1, x2, y2 = win32gui.GetWindowRect(gamewin)
        y1 += 80 # i want small region
        y2 -= 80
        x1 += 60
        x2 -= 60
        return x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)
    else :
        raise Exception('no title was given')

def get_screenshot() :
    sleep(SCRSHOT_INTV_TIME)
    squence_scrshot = np.zeros(SCRSHOT_SHAPE)
    if COLOR == 1 :
        scrshot = (screenshot(region = GAME_REGION)).convert('L').resize((SCRSHOT_W, SCRSHOT_H), resample = 0)
    elif COLOR == 3 :
        scrshot = (screenshot(region = GAME_REGION)).resize((SCRSHOT_W, SCRSHOT_H), resample = 0)
    scrshot = np.array(scrshot) / 255.5
    squence_scrshot[0, :, :] = scrshot
    return squence_scrshot
        
GAME_REGION = get_screen_rect("Getting Over It")
    
# SCREENSHOTS SETTING
SCRSHOT_W = 128
SCRSHOT_H = 72
COLOR = 3
SCRSHOT_SHAPE = (1, SCRSHOT_H, SCRSHOT_W, COLOR)
SCRSHOT_INTV_TIME = 0.004

# Q NET SETTING
Q_INPUT_SHAPE = (SCRSHOT_H, SCRSHOT_W, COLOR)
optmz = optimizers.RMSprop(lr = 0.001)

# REWARD SETTING
GAMMA = 0.1 # 1 / exp(1)
GOOD_REWARD_THRESHOLD = int(SCRSHOT_H * SCRSHOT_W * COLOR * 0.05)
GOOD_REWARD = 10.0
BAD_REWARD_MAX = -0.0
BAD_DECLINE_RATE = 0.01 # per step
BAD_REWARD_MIN = -1.0

# ACTION SETTING
MOUSE_ANGLE_DEVISION = 16
TOTAL_ACTION_NUM = MOUSE_ANGLE_DEVISION * 2
# {slow moving, fast moving}
CONTROL_PAUSE_TIME = 0.0

def ActionSet():
    actionSet = []
    for i in range(TOTAL_ACTION_NUM) :
        action_onehot = np.zeros((1, TOTAL_ACTION_NUM))
        action_onehot[0, i] = 1
        actionSet.append(action_onehot)
    return actionSet
A = ActionSet()

# STATE QUEUE SETTING    
STATEQ_LENGTH_MAX = 1600

# TRAINING SETTING
EPSILON = 0.8
MIN_EPSILON = 0.2
EPSILON_DECAY = 0.9
EPOCHES = 10
STEP_PER_EPOCH = 1000
STEP_PER_ASSIGN_TARGET = 200
STEP_PER_TRAIN = 10
TRAIN_THRESHOLD = 100
BATCH_SIZE = 8
TEST_STEPS = 200

def do_control(id) : 
    '''
    相對方向的滑動
    未來或許可以嘗試在ActionSet中加入控制快慢和距離的選擇
    '''
    slow_distance = 2000 # pixels
    fast_distance = 800 # pixels
    slow_intv = 8 # pixels
    fast_intv = 20
    intv_time = 0.0036
    
    if id < MOUSE_ANGLE_DEVISION :
        intv, distance = slow_intv, slow_distance
    else :
        intv, distance = fast_intv, fast_distance
    
    angle = 2 * math.pi * id / MOUSE_ANGLE_DEVISION
    offset_x = math.ceil(math.cos(angle) * intv)
    offset_y = math.ceil(math.sin(angle) * intv)
    
    for i in range(distance // intv) :
        directInput.directMouse(offset_x, offset_y)
        sleep(intv_time)
