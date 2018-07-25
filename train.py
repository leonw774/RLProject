import pyautogui
import math
import random
import datetime
from time import sleep

from setting import *
from PIL import Image, ImageChops
from statequeue import StateQueue
from directinputs import Keys
from QNet import QNet

from keras import optimizers
from keras import backend as K
from keras.models import Model

directInput = Keys()

def get_screenshots() :
    squence_scrshot = np.zeros(SCRSHOTS_SHAPE)
    for i in range(SCRSHOTS_N) :
        scrshot = (pyautogui.screenshot(region = GAME_REGION)).convert('L').resize((SCRSHOTS_W, SCRSHOTS_H), resample = 0)
        #scrshot.save("test" + str(i) + ".png")
        scrshot = np.asarray(scrshot) / 255.5
        squence_scrshot[0, :, :, i] = scrshot
        sleep(SCRSHOT_INTV_TIME)
    return squence_scrshot

def do_control(control_id) : 
    '''
    相對方向的滑動
    未來或許可以嘗試在ActionSet中加入控制快慢和距離的選擇
    '''
    distance = 800 # pixels
    interval = 20 # pixels
    interval_time = 0.0036
    
    angle = 2 * math.pi * control_id / MOUSE_ANGLE_DEVISION
    offset_x = math.ceil(math.cos(angle) * interval)
    offset_y = math.ceil(math.sin(angle) * interval)
    
    for i in range(distance // interval) :
        directInput.directMouse(offset_x, offset_y)
        sleep(interval_time)
    sleep(CONTROL_PAUSE_TIME)

# Action Set
actionSet = []
for i in range(TOTAL_ACTION_NUM) :
    action_onehot = np.zeros((1, TOTAL_ACTION_NUM))
    action_onehot[0, i] = 1
    actionSet.append(action_onehot)

# Q Net
Q = QNet(Q_INPUT_SHAPE, TOTAL_ACTION_NUM)
Q.compile(loss = "mse", optimizer = "RMSprop")
Q_target = QNet(Q_INPUT_SHAPE, TOTAL_ACTION_NUM)
Q_target.set_weights(Q.get_weights())

'''
We would train Q, at time t, as:
    y_pred = Q([state_t, a_t])
    y_true = r_t + gamma * max(Q_target([state_t, for a in A]))
update Q's weight in mse
after a number of steps, copy Q to Q_target.
'''

# Start Countdown
countdown = 3
for i in range(countdown) :
    print(countdown - i)
    sleep(1.0)

# print out settings
print("GOOD_REWARD_THRESHOLD: ", GOOD_REWARD_THRESHOLD)

for e in range(EPOCHES) :
    
    # click "NEW GAME"
    pyautogui.click(GAME_REGION[0] + 850, GAME_REGION[1] + 300)
    sleep(7) # wait it loading...
    
    stateQueue = StateQueue()
    
    for n in range(BATCHES_LIMIT) :
        for i in range(BATCH_SIZE) :
            cur_scrshots = get_screenshots()
            if random.random() < max(MIN_EPSILON, EPSILON * (EPSILON_DECAY ** e)):
                cur_action = random.randrange(TOTAL_ACTION_NUM)
            else :
                q_values = [Q.predict([cur_scrshots, actionSet[i]]) for i in range(TOTAL_ACTION_NUM)]
                cur_action = np.argmax(q_values)

            do_control(cur_action)
            
            nxt_scrshots = get_screenshots()
            cur_reward = stateQueue.calReward(cur_scrshots[0,:,:,0], nxt_scrshots[0,:,:,0]) # use only first scrshot
            print(cur_action, ", ", cur_reward)
            
            stateQueue.addStep(cur_scrshots, actionSet[cur_action], cur_reward, nxt_scrshots)
        # end for(BATCH)
        
        # Experience Replay
        print("replaying...")
        l = stateQueue.getLength()
        train_inputs_scrshots, train_inputs_action, reward, nxt_scrshots = stateQueue.getStepsInArray(l - BATCH_SIZE, l)
        next_reward = np.zeros((BATCH_SIZE, 1))
        
        for i in range(BATCH_SIZE) :
            next_reward[i] = max([Q_target.predict([np.reshape(nxt_scrshots[i], SCRSHOTS_SHAPE), actionSet[j]]) for j in range(TOTAL_ACTION_NUM)])
        train_targets = np.add(np.reshape(reward, (BATCH_SIZE, 1)), np.multiply(GAMMA, next_reward))
        #print("train_targets:\n", train_targets.tolist())
        
        loss = Q.train_on_batch([train_inputs_scrshots, train_inputs_action], train_targets)
        print("loss: ", loss)
    # end for(BATCHES_LIMIT)
    
    del stateQueue
    Q_target.set_weights(Q.get_weights())
    
    # Restart Game...
    # push ESC
    directInput.directKey("ESC")
    sleep(0.5)
    # click "QUIT"
    pyautogui.click(GAME_REGION[0] + 280, GAME_REGION[1] + 700)
    sleep(6.5) # wait it loading...
    # click "NEW GAME"
    pyautogui.click(GAME_REGION[0] + 850, GAME_REGION[1] + 300)
    sleep(6.5) # wait it loading...
    
    # Evaluate
    if (e % 4 == 3):
        for i in range(TEST_STEPS) :
            cur_scrshots = get_screenshots()
            q_values = [Q.predict([cur_scrshots, actionSet[i]]) for i in range(TOTAL_ACTION_NUM)]
            cur_action = np.argmax(q_values)
            do_control(cur_action)
    
# end for(EPOCHES)


    

