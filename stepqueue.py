import numpy as np
from PIL import Image
from setting import Setting as set

class StepQueue() :

    def __init__(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.actionsOccurrence = np.zeros(set.actions_num)
    
    def addStep(self, scrshot, action, reward) :
        if len(self.scrshotList) + 1 == set.stepQueue_length_max :
            self.scrshotList.pop(0)
            self.actionList.pop(0)
            self.rewardList.pop(0)
        
        if scrshot.shape != set.shot_shape:
            print("scrshot shape no good: Received", scrshot.shape, " but ", set.shot_shape, " is expected.")
            return
        
        self.scrshotList.append(scrshot[0]) # np array
        self.actionList.append(int(action)) # int
        self.rewardList.append(reward) # np array (QNet's out)
        self.actionsOccurrence[action] += 1 # record occurrence of actions
    
    def clear(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.actionsOccurrence = np.zeros(set.actions_num)
        
    def getLength(self) :
        return len(self.scrshotList)
    
    def getStepsAsArray(self, beg, size = 1) :
        to = beg + size
        return np.array(self.scrshotList[beg:to]), np.array(self.actionList[beg:to]), np.array(self.rewardList[beg:to])
        
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
    
    def getActionsOccurrence(self) :
        return self.actionsOccurrence
    
    
        