import os
import numpy as np
from win32 import win32gui

def sorting_filename_as_int(element) :
    return int(element[0 :- 4])
    
def get_game_region(title = None) :
    if title :
        gamewin = win32gui.FindWindow(None, title)
        if not gamewin:
            raise Exception('window title not found')
        #get the bounding box of the window
        x1, y1, x2, y2 = win32gui.GetWindowRect(gamewin)
        
        y1 += 30 # get rid of window bar
        
        h_padding = (y2 - y1) * 0.1
        w_padding = (x2 - x1) * 0.1
        
        y1 += h_padding
        y2 -= h_padding
        x1 += w_padding
        x2 -= w_padding
        
        return (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1))
    else :
        raise Exception("no window title was given.")
# end get_game_region

class Configuration() :
    
    # SCREENSHOTS SETTING
    shot_w = 108
    shot_h = 72
    shot_c = 3
    shot_shape = (1, shot_h, shot_w, shot_c)
    shot_resize = (shot_w, shot_h)
    shot_intv_time = 0.01
    shot_wait_max = 100
    noise_range = 0.008

    # MODEL SETTING
    use_model_name = "AC" # "QNET"
    model_input_shape = (shot_h, shot_w, shot_c)
    learning_rate = 0.001
    learning_rate_decay = 0.0
    
    # REWARD SETTING
    reward_func_name = "PixelDiffReward"
    map_path = "map/"
    map_name_list = sorted(os.listdir(map_path), key = sorting_filename_as_int)
    diff_thrshld = shot_h * shot_w * shot_c * 0.05 * ((shot_c - 1) * 0.1 + 1)
    gamma = 0.5
    base_reward = 1.0
    
    # ACTION SETTIN
    mouse_straight_angles = 8
    mouse_round_angles = 4
    actions_num = (mouse_straight_angles * 2) + (mouse_round_angles * 2)
    # ROUND ACTION ONLY HAS CLOCKWISE BECAUSE COUNTER-CLOCKWISE IS USELESS
    # [slow straight, fast straight, cwise round slow and fast, ccwise round slow and fast]
    control_intv_time = 0.001
    control_pause = 0.02

    # STEP QUEUE SETTING
    stepQueue_length_max = 500 # set 0 to be no limit

    # TRAINING SETTING
    use_target_model = False
    epsilon = 1.0
    eps_min = 0.1
    eps_decay = 0.99
    
    check_stuck = True
    stuck_thrshld = 100
    
    episodes = 10
    steps_episode = 20
    train_thrshld = 10
    steps_train = 1
    train_size = 8
    
    test_intv = 5
    draw_fig_intv = 2
    
    eps_test = 0.1
    steps_test = 100