import sys
import numpy as np
from PIL import Image
from setting import TrainingSetting as set

class StepQueue() :

    def __init__(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.nxtScrshotsList = []
        self.actionsOccurrence = np.zeros(set.actions_num)
        self.mapList = []
        
        for filename in set.mapname_list :
            if set.shot_c == 1 :
                map = Image.open("map/" + filename).convert('L').resize(set.shot_resize, resample = Image.NEAREST)
                array_map = np.reshape(np.array(map) / 255.5, (set.shot_h, set.shot_w, 1))
            elif set.shot_c == 3 :
                map = Image.open("map/" + filename).convert('RGB').resize(set.shot_resize, resample = Image.NEAREST)
                array_map = np.array(map) / 255.5
            self.mapList.append(array_map)
        
        self.r_per_map = set.total_r / len(self.mapList)
        self.r_incline_rate = (1 / set.gamma) ** (1 / len(self.mapList))
        
        print("no_move, move:", set.no_move_thrshld, set.move_much_thrshld)
        print("r_per_map:", self.r_per_map)
        print("_incline_rate:", self.r_incline_rate)
    
    def addStep(self, scrshot, action, reward, nxt_scrshot) :
        if len(self.scrshotList) + 1 == set.stepQueue_length_max :
            self.scrshotList.pop(0)
            self.actionList.pop(0)
            self.rewardList.pop(0)
            self.nxtScrshotsList.pop(0)
        
        if (scrshot.shape != set.shot_shape) or (nxt_scrshot.shape != set.shot_shape) :
            print("scrshot shape no good: Received", scrshot.shape, " but ", set.shot_shape, " is expected.")
            return
        
        self.scrshotList.append(scrshot[0]) # np array
        self.actionList.append(int(action)) # int
        self.rewardList.append(reward) # np array (QNet's out)
        self.nxtScrshotsList.append(nxt_scrshot[0]) # np array
        self.actionsOccurrence[action] += 1 # record occurrence of actions
    
    def clear(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.nxtScrshotsList = []
        self.actionsOccurrence = np.zeros(set.actions_num)
        
    def getLength(self) :
        return len(self.scrshotList)
    
    def getStepsAsArray(self, beg, size = 1) :
        to = beg + size
        return np.array(self.scrshotList[beg:to]), np.array(self.actionList[beg:to]), np.array(self.rewardList[beg:to]), np.array(self.nxtScrshotsList[beg:to])
        
    def getShotsAsArray(self, beg, size = 1) :
        try :
            return np.array(self.scrshotList[beg : beg + size])
        except :
            print("Out of Boundary Error")
    
    def getActionAsArray(self, beg, size = 1) :
        try :
            return np.array(self.actionList[beg : beg + size])
        except :
            print("Out of Boundary Error")
    
    def getRewardAsArray(self, beg, size = 1) :
        try :
            return np.array(self.rewardList[beg : beg + size])
        except :
            print("Out of Boundary Error")
    
    def getNxtShotsAsArray(self, beg, size = 1) :
        try :
            return np.array(self.nxtScrshotsList[beg : beg + size])
        except :
            print("Out of Boundary Error")
    
    def getActionsOccurrence(self) :
        return self.actionsOccurrence
        
    def getCurMap(self, scrshot) :
        min_diff = 2147483648
        for this_mapnum, this_mapshot in enumerate(self.mapList) :
            if np.sum(np.absolute(this_mapshot - scrshot)) <= min_diff :
                min_diff = d
                if this_mapnum > at_map : at_map = this_mapnum
        return at_map
    
    def isStuck(self, cur_scrshot) :
        OH_NO_YOURE_STUCK = 0
        STUCK_COUNTDOWN = set.stuck_countdown
        
        for this_map, this_scrshot in enumerate(reversed(self.scrshotList[-STUCK_COUNTDOWN:])) :
            if np.sum(np.absolute(this_scrshot - cur_scrshot)) <= set.no_move_thrshld and STUCK_COUNTDOWN > 0 :
                OH_NO_YOURE_STUCK += 1
            else :
                OH_NO_YOURE_STUCK -= 1

        return OH_NO_YOURE_STUCK > set.stuck_thrshld
    
    def calReward(self, pre_scrshot, cur_scrshot) :
        pre_scrshot = pre_scrshot[0] # before action
        cur_scrshot = cur_scrshot[0] # after action
        
        #print(cur_scrshot.shape)
        #print(self.scrshotList[0].shape)
        
        min_pre_diff = 2147483648
        pre_map = -1

        min_cur_diff = 2147483648
        cur_map = -1
        
        if self.isStuck(cur_scrshot) == True :
            return "stuck"
        
        #if np.sum(np.absolute(pre_scrshot - cur_scrshot)) < set.no_move_thrshld : return 0.0
        
        for this_mapnum, this_mapshot in enumerate(self.mapList) :
            d = np.sum(np.absolute(this_mapshot - pre_scrshot))
            if d <= min_pre_diff :
                min_pre_diff = d
                if this_mapnum > pre_map : pre_map = this_mapnum
            
            d = np.sum(np.absolute(this_mapshot - cur_scrshot))
            if d <= min_cur_diff :
                min_cur_diff = d
                if this_mapnum > cur_map : cur_map = this_mapnum
        
        #print(pre_map, "with", min_pre_diff, "to", cur_map, "with", min_cur_diff)
        
        #if min_cur_diff >= set.move_much_thrshld * 2 :
            # not in the map!?
        #    return 0.0
        
        if pre_map == cur_map :
            return 0
        else :
            pre_score = (self.r_incline_rate * pre_map) ** self.r_incline_rate
            cur_score = (self.r_incline_rate * cur_map) ** self.r_incline_rate
            return (cur_score - pre_score)
        