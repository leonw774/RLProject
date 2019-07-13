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
    learning_rate = 0.00025
    learning_rate_decay = 0.0
    
    # REWARD SETTING
    use_reward = 1
    mapname_list = sorted(os.listdir("map/"), key = sorting_filename_as_int)
    no_move_thrshld = shot_h * shot_w * shot_c * 0.056 * ((shot_c - 1) * 0.1 + 1)
    gamma = 0.5
    base_reward = 1.0
   

    # ACTION SETTIN
    mouse_straight_angles = 12
    mouse_round_angles = 6
    actions_num = (mouse_straight_angles * 2) + (mouse_round_angles * 2)
    # ROUND ACTION ONLY HAS CLOCKWISE BECAUSE COUNTER-CLOCKWISE IS USELESS
    # {slow straight(12), fast straight(12), cwise round slow and fast(12), ccwise round slow and fast(12)}
    control_pause = 0.02

    # STEP QUEUE SETTING
    stepQueue_length_max = 500 # set 0 to be no limit

    # TRAINING SETTING
    use_target_model = False
    epsilon = 1.0
    eps_min = 0.1
    eps_decay = 0.9
        
    check_stuck = True
    stuck_thrshld = 100
    
    episodes = 40
    steps_episode = 200
    train_thrshld = 50
    steps_train = 1
    train_size = 32
    
    test_intv = 5
    draw_fig_intv = 20
    
    eps_test = 0.1
    steps_test = 100
    
    