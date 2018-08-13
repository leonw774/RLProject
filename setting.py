import numpy as np
from win32 import win32gui
from keras import optimizers

def get_game_region(title) :
    if title :
        gamewin = win32gui.FindWindow(None, title)
        if not gamewin:
            raise Exception('window title not found')
        #get the bounding box of the window
        x1, y1, x2, y2 = win32gui.GetWindowRect(gamewin)
        
        y1 += 50 # get rid of window bar
        y2 -= 10
        x1 += 10
        x2 -= 10
        
        return (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1))
    else :
        raise Exception("no window title was given.")
        
GameRegion = get_game_region("Getting Over It")

class TrainingSetting() :
    
    # SCREENSHOTS SETTING
    scrshot_w = 128
    scrshot_h = 72
    color_size = 3
    scrshot_shape = (1, scrshot_h, scrshot_w, color_size)
    scrshot_resize = (scrshot_w, scrshot_h)
    scrshot_intv_time = 0.006

    # Q NET SETTING
    model_input_shape = (scrshot_h, scrshot_w, color_size)
    model_optimizer = optimizers.RMSprop(lr = 0.0002)

    # REWARD SETTING
    gamma = 0.9 # 1 / exp(1)
    good_r_thrshld = int(scrshot_h * scrshot_w * color_size * 0.075)
    good_r = 10.0
    bad_r_max = -0.0
    bad_decline_rate = 0.1 # per step
    bad_r_min = -10.0

    # ACTION SETTING
    mouse_angle_devision = 16
    actions_num = mouse_angle_devision * 2
    # {slow moving, fast moving}
    do_control_pause = 0.003

    # STATE QUEUE SETTING    
    statequeue_length_max = 1000

    # TRAINING SETTING
    epsilon = 1.0
    eps_min = 0.1
    eps_decay = 0.99
    epoches = 100
    steps_epoch = 1000
    train_thrshld = 80
    steps_train = 8
    batch_size = 16
    steps_update_target = 100
    
    eps_test = 0.01
    steps_test = 400
>>>>>>> 8f179155e84ba60564f599fd5d6ddb2198715fa3
    