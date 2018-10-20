import numpy as np
from win32 import win32gui

class TrainingSetting() :
    
    # SCREENSHOTS SETTING
    shot_n = 2
    shot_w = 128
    shot_h = 72
    shot_c = 1
    shot_shape = (1, shot_h, shot_w, shot_c)
    shot_resize = (shot_w, shot_h)
    shot_intv_time = 0.01
    noise_range = 0.03
    
    def get_game_region(title = None) :
        if title :
            gamewin = win32gui.FindWindow(None, title)
            if not gamewin:
                raise Exception('window title not found')
            #get the bounding box of the window
            x1, y1, x2, y2 = win32gui.GetWindowRect(gamewin)
            
            h_padding = (y2 - y1) * 0.1
            w_padding = (x2 - x1) * 0.1
            
            y1 += h_padding # get rid of window bar
            y2 -= h_padding
            x1 += w_padding
            x2 -= w_padding
            
            return (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1))
        else :
            raise Exception("no window title was given.")
    # end get_game_region

    # Q NET SETTING
    model_input_shape = (shot_h, shot_w, shot_c * shot_n)

    # REWARD SETTING
    gamma = 0.36787944117 # 1 / exp(1)
    good_thrshld = shot_h * shot_w * shot_c * (0.03 + 2 * noise_range) # 0.09
    no_move_thrshld = shot_h * shot_w * shot_c * 0.03
    
    stuck_countdown = 75
    stuck_thrshld = 50
    
    total_r = 100.0
    good_r = 1.0
    bad_r = 0.0
    decline_rate = 0.8 # 0.0 ~ 1.0, per step

    # ACTION SETTING
    mouse_angle_devision = 16
    actions_num = mouse_angle_devision * 2
    # {slow moving, fast moving}
    do_control_pause = 0.1

    # STEP QUEUE SETTING
    stepQueue_length_max = 10000 # set 0 to be no limit

    # TRAINING SETTING
    epsilon = 1.0
    eps_min = 0.2
    eps_decay = 0.994
    epoches = 200
    steps_epoch = 500
    train_thrshld = 101
    steps_train = 6
    train_size = 64
    steps_update_target = 100 # set to 0 to disable
    
    no_reward_break = False
    
    eps_test = 0.01
    steps_test = 50
    