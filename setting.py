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
    color_size = 1
    scrshot_shape = (1, scrshot_h, scrshot_w, color_size)
    scrshot_resize = (scrshot_w, scrshot_h)
    scrshot_intv_time = 0.006
    noise_range = 0.03

    # Q NET SETTING
    model_input_shape = (scrshot_h, scrshot_w, color_size)
    model_optimizer = optimizers.Adam(lr = 0.001, decay = 0.999)

    # REWARD SETTING
    gamma = 0.26894142137 # 1 / (1 + exp(1))
    good_r_thrshld = scrshot_h * scrshot_w * color_size * (0.02 + 2 * noise_range) # 0.08
    no_move_thrshld = scrshot_h * scrshot_w * color_size * 0.01
    good_r = 10.0
    bad_r_max = -0.0
    bad_decline_rate = 1.0 # per step
    bad_r_min = -1.0

    # ACTION SETTING
    mouse_angle_devision = 18
    actions_num = mouse_angle_devision * 2
    # {slow moving, fast moving}
    do_control_pause = 0.006

    # STATE QUEUE SETTING
    
    statequeue_length_max = 10000 # set 0 to be no limit

    # TRAINING SETTING
    epsilon = 1.0
    eps_min = 0.2
    eps_decay = 0.999
    epoches = 400
    steps_epoch = 2501
    train_thrshld = 200
    steps_train = 8
    train_size = 64
    #batch_size = 8
    steps_update_target = 100 # set to 0 to disable
    
    eps_test = 0.02
    steps_test = 400
    