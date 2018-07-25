import numpy as np
from setting import *

class StateQueue() :

    def __init__(self) :
        self.scrshotsList = [] 
        self.actionList = []
        self.rewardList = []
        self.nxtScrshotsList = []
    
    def addStep(self, scrshots, action, reward, nxt_scrshots) :
        if (scrshots.shape != SCRSHOTS_SHAPE) or (nxt_scrshots.shape != SCRSHOTS_SHAPE) :
            print("scrshots shape no good: Received", scrshots.shape, " but ", SCRSHOTS_SHAPE, " is expected.")
            return
        self.scrshotsList.append(scrshots[0])
        self.actionList.append(action[0])
        self.rewardList.append(reward)
        self.nxtScrshotsList.append(nxt_scrshots[0])
        
    def getLength(self) :
        return len(self.scrshotsList)
            
    def getStepsInArray(self, begin, to) :
        return np.array(self.scrshotsList[begin:to]), np.array(self.actionList[begin:to]), np.array(self.rewardList[begin:to]), np.array(self.nxtScrshotsList[begin:to])
        
    def getScrshotAt(self, stepNum) :
        try :
            return scrshotsList[stepNum]
        except :
            raise("Out of Boundary Error")
    
    def getActionAt(self, stepNum) :
        try :
            return actionList[stepNum]
        except :
            raise("Out of Boundary Error")
    
    def getRewardAt(self, stepNum) :
        try :
            return rewardList[stepNum]
        except :
            raise("Out of Boundary Error")
    
    def getNxtScrshotAt(self, stepNum) :
        try :
            return nxtScrshotsList[stepNum]
        except :
            raise("Out of Boundary Error")
                      
    def calReward(self, pre_scrshot, cur_scrshot) :
        if cur_scrshot.shape != SCRSHOTS_SHAPE[1:3] :
            print("cur_scrshot shape no good: Received", cur_scrshot.shape, " but ", SCRSHOTS_SHAPE[1:3], " is expected.")
            return
        
        cur_diff = 2147483648
        cur_steps_before = -1
        pre_diff = 2147483648
        pre_steps_before = -1
        
        for step, scrshot_seq in enumerate(self.scrshotsList) :
            cur_d = np.sum(np.absolute(np.subtract(scrshot_seq[:, :, 0], cur_scrshot)))
            pre_d = np.sum(np.absolute(np.subtract(scrshot_seq[:, :, 0], pre_scrshot)))
            if cur_d < cur_diff :
                cur_diff = cur_d
                cur_steps_before = len(self.scrshotsList) - step
            if pre_d < cur_diff :
                pre_diff = cur_d
                pre_steps_before = len(self.scrshotsList) - step
        #print("cur_diff", cur_diff)
        
        OH_NO_YOURE_NOT_MOVING = cur_steps_before < 3 and pre_steps_before < 3
        FIND_NEW_ENV = cur_diff > GOOD_REWARD_THRESHOLD
        GOING_FORWARD = pre_steps_before < cur_steps_before
        
        diff_score = cur_diff / GOOD_REWARD_THRESHOLD * 0.05 * GOOD_REWARD
        #print("diff_score:", diff_score)
        if diff_score > GOOD_REWARD : diff_score = GOOD_REWARD
        
        if FIND_NEW_ENV or cur_steps_before == -1 :
            if not OH_NO_YOURE_NOT_MOVING :
                #print("Find new env")
                return GOOD_REWARD + diff_score
            else :
                #print("Find new env, but still YOURE NOT MOVING")
                return GOOD_REWARD - BAD_DECLINE_RATE * cur_steps_before
        else :
            if GOING_FORWARD :
                #print("is going forward")
                return diff_score
            elif OH_NO_YOURE_NOT_MOVING :
                #print("NO YOURE NOT MOVING")
                return BAD_REWARD_MAX
            else :
                #print("YOU SCREW")
                bad = BAD_REWARD_MAX - BAD_DECLINE_RATE * cur_steps_beforebad
                return  if bad > BAD_REWARD_MIN else BAD_REWARD_MIN
    