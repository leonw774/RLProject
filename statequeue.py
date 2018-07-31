import numpy as np
from setting import TrainingSetting as set

class StateQueue() :

    def __init__(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.nxtScrshotsList = []
    
    def addStep(self, scrshot, action, reward, nxt_scrshot) :
        if len(self.scrshotList) == set.statequeue_length_max :
            self.scrshotList.pop(0)
            self.actionList.pop(0)
            self.rewardList.pop(0)
            self.nxtScrshotsList.pop(0)
            
        if (scrshot.shape != set.scrshot_shape) or (nxt_scrshot.shape != set.scrshot_shape) :
            print("scrshot shape no good: Received", scrshot.shape, " but ", set.scrshot_shape, " is expected.")
            return
        self.scrshotList.append(scrshot[0])
        self.actionList.append(action[0])
        self.rewardList.append(reward)
        self.nxtScrshotsList.append(nxt_scrshot[0])
        
    def getLength(self) :
        return len(self.scrshotList)
            
    def getStepsInArray(self, begin, to) :
        return np.array(self.scrshotList[begin:to]), np.array(self.actionList[begin:to]), np.array(self.rewardList[begin:to]), np.array(self.nxtScrshotsList[begin:to])
        
    def getScrshotAt(self, stepNum) :
        try :
            return scrshotList[stepNum]
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
        pre_scrshot = pre_scrshot[0]
        cur_scrshot = cur_scrshot[0]
        if len(self.scrshotList) <= 2 : return 0
        
        cur_diff = 2147483648
        cur_steps_before = -1
        pre_diff = 2147483648
        pre_steps_before = -1
        
        OH_NO_YOURE_NOT_MOVING = (np.sum(np.absolute(np.subtract(pre_scrshot, cur_scrshot))) < (set.good_r // 2))
        
        if not OH_NO_YOURE_NOT_MOVING :
            for step, scrshot_seq in enumerate(self.scrshotList) :
                cur_d = np.sum(np.absolute(np.subtract(scrshot_seq, cur_scrshot)))
                pre_d = np.sum(np.absolute(np.subtract(scrshot_seq, pre_scrshot)))
                if cur_d < cur_diff :
                    cur_diff = cur_d
                    cur_steps_before = len(self.scrshotList) - step
                if pre_d < cur_diff :
                    pre_diff = cur_d
                    pre_steps_before = len(self.scrshotList) - step
            #print("cur_diff", cur_diff)
            OH_NO_YOURE_NOT_MOVING = cur_steps_before < 3 and pre_steps_before < 3
        
        diff_score = set.good_r * cur_diff / set.good_r_thrshld // 2
        if diff_score > set.good_r : diff_score = set.good_r
        elif diff_score < 0 : diff_score = 0
        #print("diff_score", diff_score)
        
        if cur_diff > set.good_r_thrshld or cur_steps_before == -1 :
            #print("Find new env")
            return set.good_r
        else :
            if not OH_NO_YOURE_NOT_MOVING :
                #print("is moving")
                bad = diff_score - set.bad_decline_rate * cur_steps_before
                return bad if bad > set.bad_r_min else set.bad_r_min
            else :
                return set.bad_r_max
            '''
                #print("YOU SCREW")
                bad = bad_r_max - bad_decline_rate * cur_steps_before
                return bad if bad > bad_r_min else bad_r_min
            '''
    