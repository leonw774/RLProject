import numpy as np
from keras import optimizers

class TrainingSetting() :
    
    # SCREENSHOTS SETTING
    scrshot_n = 2
    scrshot_w = 128
    scrshot_h = 72
    color_size = 1
    scrshot_shape = (1, scrshot_h, scrshot_w, color_size)
    scrshot_resize = (scrshot_w, scrshot_h)
    scrshot_intv_time = 0.006
    noise_range = 0.03

    # Q NET SETTING
    model_input_shape = (scrshot_h, scrshot_w, color_size * scrshot_n)
    model_optimizer = optimizers.Adam(lr = 0.001, decay = 0.999)

    # REWARD SETTING
    gamma = 0.26894142137 # 1 / (1 + exp(1))
    good_thrshld = scrshot_h * scrshot_w * color_size * (0.0396 + 2 * noise_range) # 0.1
    no_move_thrshld = scrshot_h * scrshot_w * color_size * 0.0396
    
    stuck_countdown = 100
    stuck_thrshld = 50
    
    use_compare_block = False
    block_side_num = 4
    block_num = block_side_num**2
    compare_stride = max(scrshot_h // block_side_num, scrshot_w // block_side_num)
    compare_block_size = (block_num, scrshot_h // block_side_num, scrshot_w // block_side_num, color_size)
    
    good_r = 1.0
    bad_r = -0.0
    bad_decline_rate = 0.01 # per step

    # ACTION SETTING
    mouse_angle_devision = 18
    actions_num = mouse_angle_devision * 2
    # {slow moving, fast moving}
    do_control_pause = 0.1

    # STEP QUEUE SETTING
    stepQueue_length_max = 10000 # set 0 to be no limit

    # TRAINING SETTING
    epsilon = 1.0
    eps_min = 0.2
    eps_decay = 0.999
    epoches = 100
    steps_epoch = 2000
    train_thrshld = 201
    steps_train = 8
    train_size = 64
    #batch_size = 8
    steps_update_target = 200 # set to 0 to disable
    
    eps_test = 0.02
    steps_test = 400
    