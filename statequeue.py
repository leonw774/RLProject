import numpy as np
from setting import TrainingSetting as set

class StateQueue() :

    def __init__(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.nxtScrshotsList = []
    
    def addStep(self, scrshot, action, reward, nxt_scrshot) :
        if len(self.scrshotList) + 1 == set.statequeue_length_max :
            self.scrshotList.pop(0)
            self.actionList.pop(0)
            self.rewardList.pop(0)
            self.nxtScrshotsList.pop(0)
        '''
        if (scrshot.shape != set.scrshot_shape) or (nxt_scrshot.shape != set.scrshot_shape) :
            print("scrshot shape no good: Received", scrshot.shape, " but ", set.scrshot_shape, " is expected.")
            return
        '''
        self.scrshotList.append(scrshot[0]) # np array
        self.actionList.append(action) # int
        self.rewardList.append(reward) # float
        self.nxtScrshotsList.append(nxt_scrshot[0]) # np array
        
    def getLength(self) :
        return len(self.scrshotList)
            
    def getStepsAsArray(self, begin, size) :
        to = begin + size
        return np.array(self.scrshotList[begin:to]), np.array(self.actionList[begin:to]), np.array(self.rewardList[begin:to]), np.array(self.nxtScrshotsList[begin:to])
        
    def getScrshotAsArray(self, beg, size) :
        try :
            return np.array(self.scrshotList[beg : beg + size])
        except :
            print("Out of Boundary Error")
    
    def getActionAsArray(self, beg, size) :
        try :
            return np.array(self.actionList[beg : beg + size])
        except :
            print("Out of Boundary Error")
    
    def getRewardAsArray(self, beg, size) :
        try :
            return np.array(self.rewardList[beg : beg + size])
        except :
            print("Out of Boundary Error")
    
    def getNxtScrshotAsArray(self, beg, size) :
        try :
            return np.array(self.nxtScrshotsList[beg : beg + size])
        except :
            print("Out of Boundary Error")
                      
    def calReward(self, pre_scrshot, cur_scrshot) :
        pre_scrshot = pre_scrshot[0]
        cur_scrshot = cur_scrshot[0]
        if len(self.scrshotList) <= 2 : return 0
        
        cur_diff = 2147483648
        cur_distance = -1
        
        OH_NO_YOURE_NOT_MOVING = np.sum(np.absolute(np.subtract(pre_scrshot, cur_scrshot))) < set.no_move_thrshld
        OH_NO_YOURE_STUCK = 0
        
        if not OH_NO_YOURE_NOT_MOVING :
            # find the screenshot that is most similar: smallest diff
            for step, scrshot_seq in enumerate(self.scrshotList) :
                cur_d = np.sum(np.absolute(np.subtract(scrshot_seq, cur_scrshot)))
                pre_d = np.sum(np.absolute(np.subtract(scrshot_seq, pre_scrshot)))
                
                if cur_d <= set.no_move_thrshld : OH_NO_YOURE_STUCK += 1
                else : OH_NO_YOURE_STUCK = 0
                if OH_NO_YOURE_STUCK > 32 :
                    print("stuck")
                    return "stuck"
                
                if cur_d < cur_diff :
                    cur_diff = cur_d
                    cur_distance = len(self.scrshotList) - step
            # check again if is not moving
            OH_NO_YOURE_NOT_MOVING = cur_distance < 2 
            
        if not OH_NO_YOURE_NOT_MOVING :
        
            if cur_diff > set.good_r_thrshld or cur_distance == -1 :
                #print("Find new env")
                return set.good_r
            else :
                diff_score = (cur_diff / set.good_r_thrshld) * set.good_r
                diff_score = set.good_r if diff_score > set.good_r else diff_score
                #print("diff_score", diff_score)
                bad = diff_score - set.bad_decline_rate * cur_distance
                #print("is moving", bad)
                return bad if bad > set.bad_r_max else set.bad_r_max
        else :
            #print("YOU SCREW")
            #return set.bad_r_max
            bad = set.bad_r_max - set.bad_decline_rate * cur_distance
            return bad if bad < set.bad_r_min else set.bad_r_min
    