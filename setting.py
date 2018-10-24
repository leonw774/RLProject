import os
import numpy as np
from win32 import win32gui

def sorting_filename_as_int(element) :
    return int(element[0 :- 4])

class TrainingSetting() :
    
    # SCREENSHOTS SETTING
    shot_n = 1
    shot_w = 108
    shot_h = 72
    shot_c = 3
    shot_shape = (1, shot_h, shot_w, shot_c)
    shot_resize = (shot_w, shot_h)
    shot_intv_time = 0.005
    shot_wait_max = 20
    noise_range = 0.01
    
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
    mapname_list = sorted(os.listdir("map/"), key = sorting_filename_as_int)
    gamma = 0.36787944117 # 1 / exp(1)
    move_much_thrshld = shot_h * shot_w * shot_c * (0.07 + 2 * noise_range) * ((shot_c - 1) * 0.01 + 1) # 0.09
    no_move_thrshld = shot_h * shot_w * shot_c * 0.03 * ((shot_c - 1) * 0.01 + 1)
    stuck_countdown = 125
    stuck_thrshld = 120
    total_r = len(mapname_list)

    # ACTION SETTIN
    mouse_straight_angles = 12
    mouse_round_angles = 4
    actions_num = (mouse_straight_angles + mouse_round_angles) * 2
    # {slow straight, fast straight, clockwise round, ccwise round}
    do_control_pause = 0.01

    # STEP QUEUE SETTING
    stepQueue_length_max = 10000 # set 0 to be no limit

    # TRAINING SETTING
    epsilon = 1.0
    eps_min = 0.2
    eps_decay = 0.996
    epoches = 200
    steps_epoch = 500
    train_thrshld = 80
    steps_train = 8
    train_size = 64
    steps_update_target = 100 # set to 0 to disable
    
    no_reward_break = False
    
    eps_test = 0.2
    steps_test = 500
    