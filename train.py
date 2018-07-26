import random
from time import sleep
from pyautogui import click

from setting import *
from directinputs import Keys
from statequeue import StateQueue
from QNet import QNet

from keras.models import Model

# Q Net
Q = QNet(Q_INPUT_SHAPE, TOTAL_ACTION_NUM)
Q.compile(loss = "mse", optimizer = optmz)
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
#print("GOOD_REWARD_THRESHOLD: ", GOOD_REWARD_THRESHOLD)

for e in range(EPOCHES) :
    
    # click "NEW GAME"
    click(GAME_REGION[0] + 780, GAME_REGION[1] + 220)
    sleep(7) # wait it loading...
    
    stateQueue = StateQueue()

    for n in range(STEP_PER_EPOCH) :
        cur_scrshot = get_screenshot()
        if random.random() < max(MIN_EPSILON, EPSILON * (EPSILON_DECAY ** e)):
            cur_action = random.randrange(TOTAL_ACTION_NUM)
        else :
            q_values = [Q.predict([cur_scrshot, A[a]])[0,0] for a in range(TOTAL_ACTION_NUM)]
            cur_action = np.argmax(q_values)

        do_control(cur_action)
        
        nxt_scrshot = get_screenshot()
        cur_reward = stateQueue.calReward(cur_scrshot, nxt_scrshot)
        print(cur_action, ",", cur_reward)
        
        stateQueue.addStep(cur_scrshot, A[cur_action], cur_reward, nxt_scrshot)
        
        if (stateQueue.getLength() > BATCH_SIZE
        and n > TRAIN_THRESHOLD
        and n % STEP_PER_TRAIN == 0) :
            # Experience Replay
            r = random.randint(BATCH_SIZE, stateQueue.getLength())
            inputs_scrshot, inputs_action, reward, nxt_scrshot = stateQueue.getStepsInArray(r - BATCH_SIZE, r)
            next_reward = np.zeros((BATCH_SIZE, 1))
            
            for j in range(BATCH_SIZE) :
                next_reward[j] = max([Q_target.predict([np.reshape(nxt_scrshot[j], SCRSHOT_SHAPE), A[a]]) for a in range(TOTAL_ACTION_NUM)])
            train_targets = np.add(np.reshape(reward, (BATCH_SIZE, 1)), np.multiply(GAMMA, next_reward))
            #print("train_targets:\n", train_targets.tolist())
            
            loss = Q.train_on_batch([inputs_scrshot, inputs_action], train_targets)
            print("loss: ", loss)
            
        if n % STEP_PER_ASSIGN_TARGET == 0 and n != 0 :
            Q_target.set_weights(Q.get_weights())
            
        # end for(STEP_PER_EPOCH)  
    
    del stateQueue

    # Restart Game...
    # push ESC
    directInput.directKey("ESC")
    sleep(0.1)
    # click "QUIT"
    click(GAME_REGION[0] + 220, GAME_REGION[1] + 610)
    sleep(6.9) # wait it loading
# end for(EPOCHES)

Q_target.save_weights("Q_target_weight.h5")
