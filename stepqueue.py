import numpy as np
from PIL import Image
from configure import Configuration as cfg

class StepQueue() :

    def __init__(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
    
    def add_step(self, scrshot, action, reward) :
        if len(self.scrshotList) + 1 == cfg.stepQueue_length_max :
            self.scrshotList.pop(0)
            self.actionList.pop(0)
            self.rewardList.pop(0)
        
        if scrshot.shape != cfg.shot_shape:
            print("scrshot bad shape: Received", scrshot.shape, " but ", cfg.shot_shape, " is expected.")
            return
        
        self.scrshotList.append(scrshot[0]) # np array
        self.actionList.append(action.asarray) # np array
        self.rewardList.append(reward) # np array (QNet's out)
    
    def clear(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
    
    def __len__(self) :
        return len(self.scrshotList)
    
    def get_steps(self, beg, size = 1) :
        to = beg + size
        return np.array(self.scrshotList[beg:to]), np.array(self.actionList[beg:to]), np.array(self.rewardList[beg:to])
    
        