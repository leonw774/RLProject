import numpy as np
from win32 import win32gui

def get_screen_rect(title = None):
    if title :
        gamewin = win32gui.FindWindow(None, title)
        if not gamewin:
            raise Exception('window title not found')
        #get the bounding box of the window
        x1, y1, x2, y2 = win32gui.GetWindowRect(gamewin)
        y1 += 10
        return x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)
    else :
        raise Exception('no title was given')
        
if __name__ == '__main__' :
    print("Why, don't run this in directly!")
else :
    GAME_REGION = get_screen_rect("Getting Over It")
    
# SCREENSHOTS SETTING
SCRSHOTS_W = 128
SCRSHOTS_H = 108
SCRSHOTS_N = 1
SCRSHOTS_SHAPE = (1, SCRSHOTS_H, SCRSHOTS_W, SCRSHOTS_N)
SCRSHOT_INTV_TIME = 0.002

# Q NET SETTING
Q_INPUT_SHAPE = (SCRSHOTS_H, SCRSHOTS_W, SCRSHOTS_N)

# REWARD SETTING
GAMMA = 0.05 # 1 / exp(1)
GOOD_REWARD_THRESHOLD = int(SCRSHOTS_H * SCRSHOTS_W * 0.05)
GOOD_REWARD = 10.0
BAD_REWARD_MAX = -0.0
BAD_DECLINE_RATE = 0.01 # per step
BAD_REWARD_MIN = -10.0

# ACTION SETTING
MOUSE_ANGLE_DEVISION = 18
MOUSE_ACTION_NUM = MOUSE_ANGLE_DEVISION + 1 #{angle_0, angle_1, ..., angle_n, no_angle}
KEYBOARD_ACTION_NUM = 0 #{key_0, key_1, ..., key_n}
TOTAL_ACTION_NUM = MOUSE_ACTION_NUM + KEYBOARD_ACTION_NUM
CONTROL_PAUSE_TIME = 0.001

# TRAINING SETTING
EPSILON = 0.9
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.999
EPOCHES = 40
BATCH_SIZE = 32
BATCHES_LIMIT = 10
TEST_STEPS = BATCH_SIZE * BATCHES_LIMIT
