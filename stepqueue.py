import sys
import numpy as np
from setting import TrainingSetting as set
from PIL import Image
class StepQueue() :

    def __init__(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.nxtScrshotsList = []
        self.actionsOccurrence = np.zeros(set.actions_num)
    
    def addStep(self, scrshot, action, reward, nxt_scrshot) :
        if len(self.scrshotList) + 1 == set.stepQueue_length_max :
            self.scrshotList.pop(0)
            self.actionList.pop(0)
            self.rewardList.pop(0)
            self.nxtScrshotsList.pop(0)
        '''
        if (scrshot.shape != set.shot_shape) or (nxt_scrshot.shape != set.shot_shape) :
            print("scrshot shape no good: Received", scrshot.shape, " but ", set.shot_shape, " is expected.")
            return
        '''
        self.scrshotList.append(scrshot[0]) # np array
        self.actionList.append(action) # int
        self.rewardList.append(reward) # float
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
                      
    def calReward(self, pre_scrshot, cur_scrshot,count) :
        if count < 50:
            i = count
        else:
            i = count - 20
        pre_find = False
        cur_find = False
        pre = 0.0
        cur = 0.0
        path = r"C:\Users\lin2\Documents\GitHub\QWOPProject_now\QWOP\test_scrshot"
        while i < 993 :
            file_path = path + str(i) + r".png"
            im_frame = Image.open(file_path)
            if set.color_size == 1 :
                scrshot = (im_frame.getdata()).convert('L').resize(set.shot_resize)
                np_frame = np.reshape(np.array(scrshot) / 255.5, set.shot_shape)
            elif set.color_size == 3 :
                scrshot = (im_frame.getdata()).convert('RGB').resize(set.shot_resize)
                np_frame[0] = np.array(scrshot) / 255.5
            
            #find cur
            result_arrray = cur_scrshot- np_frame
            result = np.sum(result_arrray)
            if abs(result) < 0.1:
                #print(float(i)/10,"m")
                cur = float(i)/10
                cur_find = True
            #find pre
            result_arrray = pre_scrshot- np_frame
            result = np.sum(result_arrray)
            if abs(result) < 0.1:
                #print(float(i)/10,"m")
                pre = float(i)/10
                pre_find = True
            #all find
            if pre_find == True:
                if cur_find == True:
                    print("cur",cur,"m")
                    return int(cur*10),cur - pre
            i = i + 1
        return 1,-1
