import numpy as np
import math
from pyautogui import screenshot
from PIL import Image

from time import sleep
from win32 import win32gui
from directinputs import Keys
directInput = Keys()

def get_screen_rect(title = None):
    if title :
        gamewin = win32gui.FindWindow(None, title)
        if not gamewin:
            raise Exception('window title not found')
        #get the bounding box of the window
        x1, y1, x2, y2 = win32gui.GetWindowRect(gamewin)
        y1 += 100 # i want smaller region
        y2 -= 100
        x1 += 80
        x2 -= 80
        return x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)
    else :
        raise Exception('no title was given')

def get_screenshot() :
    sleep(SCRSHOT_INTV_TIME)
    array_scrshot = np.zeros(SCRSHOT_SHAPE)
    if COLOR == 1 :
        scrshot = (screenshot(region = GAME_REGION)).convert('L').resize((SCRSHOT_W, SCRSHOT_H), resample = Image.NEAREST)
    elif COLOR == 3 :
        scrshot = (screenshot(region = GAME_REGION)).convert('RGB').resize((SCRSHOT_W, SCRSHOT_H), resample = Image.NEAREST)
    array_scrshot[0] = np.array(scrshot) / 255.5
    return array_scrshot
    
def add_noise(noisy_scrshot) :
    noisy_scrshot += np.random.uniform(low = -0.01, high = 0.01, size = SCRSHOT_SHAPE)
    noisy_scrshot[noisy_scrshot > 1.0] = 1.0
    noisy_scrshot[noisy_scrshot < 0.0] = 0.0
    return noisy_scrshot
        
GAME_REGION = get_screen_rect("Getting Over It")
    
# SCREENSHOTS SETTING
SCRSHOT_W = 128
SCRSHOT_H = 72
COLOR = 3
SCRSHOT_SHAPE = (1, SCRSHOT_H, SCRSHOT_W, COLOR)
SCRSHOT_INTV_TIME = 0.006

# Q NET SETTING
Q_INPUT_SHAPE = (SCRSHOT_H, SCRSHOT_W, COLOR)

# REWARD SETTING
GAMMA = 0.1 # 1 / exp(1)
GOOD_REWARD_THRESHOLD = int(SCRSHOT_H * SCRSHOT_W * COLOR * 0.06)
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
EPSILON = 0.87
MIN_EPSILON = 0.2
EPSILON_DECAY = 0.975
EPOCHES = 20
STEP_PER_EPOCH = 1000
STEP_PER_ASSIGN_TARGET = 200
STEP_PER_TRAIN = 4
TRAIN_THRESHOLD = 100
BATCH_SIZE = 16
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
