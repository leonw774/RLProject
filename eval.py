import random
from pyautogui import click
from keras.models import Model

from setting import *
from statequeue import StateQueue
from QNet import QNet

Q_target = QNet(Q_INPUT_SHAPE, TOTAL_ACTION_NUM)
Q_target.load_weights("Q_target_weight.h5")

# Evaluate
print("Evaluate")
# click "NEW GAME"
click(GAME_REGION[0] + 850, GAME_REGION[1] + 290)
sleep(7) # wait it loading...
stateQueue = StateQueue()
for i in range(TEST_STEPS) :
    cur_scrshot = get_screenshot()
    if random.random() < MIN_EPSILON :
        cur_action = random.randrange(TOTAL_ACTION_NUM)
    else :
        q_values = [Q_target.predict([cur_scrshot, A[i]])[0,0] for i in range(TOTAL_ACTION_NUM)]
        #print(q_values)
        cur_action = np.argmax(q_values)
    do_control(cur_action)
    nxt_scrshot = get_screenshot()
    stateQueue.calReward(cur_scrshot, nxt_scrshot)
    cur_reward = stateQueue.calReward(cur_scrshot, nxt_scrshot)
    stateQueue.addStep(cur_scrshot, A[cur_action], cur_reward, nxt_scrshot)
    print(cur_action, ", ", cur_reward)