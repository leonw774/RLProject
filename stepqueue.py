import numpy as np
from setting import TrainingSetting as set

class StepQueue() :

    def __init__(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.nxtScrshotsList = []
    
    def addStep(self, scrshot, action, reward, nxt_scrshot) :
        if len(self.scrshotList) + 1 == set.stepQueue_length_max :
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
    
    def clear(self) :
        self.scrshotList = [] 
        self.actionList = []
        self.rewardList = []
        self.nxtScrshotsList = []
        
    def getLength(self) :
        return len(self.scrshotList)
            
    def getStepsAsArray(self, beg, size = 1) :
        to = beg + size
        return np.array(self.scrshotList[beg:to]), np.array(self.actionList[beg:to]), np.array(self.rewardList[beg:to]), np.array(self.nxtScrshotsList[beg:to])
        
    def getScrshotAsArray(self, beg, size = 1) :
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
    
    def getNxtScrshotAsArray(self, beg, size = 1) :
        try :
            return np.array(self.nxtScrshotsList[beg : beg + size])
        except :
            print("Out of Boundary Error")
                      
    def calReward(self, pre_scrshot, cur_scrshot) :
        pre_scrshot = pre_scrshot[0] # before action
        cur_scrshot = cur_scrshot[0] # after action
        if len(self.scrshotList) <= 2 : return 0
        
        #print(cur_scrshot.shape)
        #print(self.scrshotList[0].shape)
        
        min_full_diff = 2147483648
        min_full_diff_dist = -1
        min_block_diff = 2147483648
        min_block_diff_dist = -1
        
        full_score = -1
        block_score = -1
        
        OH_NO_YOURE_NOT_MOVING = np.sum(np.absolute(pre_scrshot - cur_scrshot)) < set.no_move_thrshld
        OH_NO_YOURE_STUCK = 0
        NOT_STUCK_COUNTDOWN = set.stuck_countdown
        
        if set.use_compare_block :
            block_h = set.compare_block_size[1]
            block_w = set.compare_block_size[2]
            compare_blocks = np.zeros(set.compare_block_size)
           
        # find the screenshot that is most similar: smallest diff
        for this_step, this_scrshot in enumerate(reversed(self.scrshotList)) :
            # full image diff
            d = np.sum(np.absolute(np.subtract(this_scrshot, cur_scrshot)))
            if d < min_full_diff :
                min_full_diff = d
                min_full_diff_dist = len(self.scrshotList) - this_step
            
            # stuck check
            if d <= set.no_move_thrshld and NOT_STUCK_COUNTDOWN > 0 :
                OH_NO_YOURE_STUCK += 1  
            else : 
                OH_NO_YOURE_STUCK = 0
            NOT_STUCK_COUNTDOWN -= 1
            
            if OH_NO_YOURE_STUCK > set.stuck_thrshld :
                print("stuck")
                return "stuck"
            
            # blocks image diff 
            # this algorithm is bigO(n^3), very slow. Only use if CPU is good
            if set.use_compare_block and d > set.no_move_thrshld and not OH_NO_YOURE_NOT_MOVING :
                # make compare_blocks
                # hack from stackoverflow :
                compare_blocks = cur_scrshot.reshape(block_h, set.block_side_num, -1, set.block_side_num).swapaxes(1,2).reshape(-1, set.block_side_num, set.block_side_num)
                
                compare_result_array = np.full((set.block_num), 2147483648)
                for n in range(set.block_num) :
                    i = 0
                    j = 0
                    while(i+block_h < set.scrshot_h) :
                        while(j+block_w < set.scrshot_w) :
                            compare_result_array[n] = min(
                                compare_result_array[n],
                                np.sum(
                                    np.absolute(
                                        this_scrshot[i:i+block_h, j:j+block_w] - compare_blocks[n]
                                    )
                                )
                            )
                            j += set.compare_stride
                        i += set.compare_stride
                # compare_result :
                # 0 --> there is same (diff smaller then thresold)
                # 1 --> there is no same (diff larger)
                block_diff = np.sum((compare_result_array > set.good_thrshld / set.block_num).astype(np.int))
                if block_diff < min_block_diff :
                    min_block_diff = block_diff
                    min_block_diff_dist = len(self.scrshotList) - this_step
                    
                # check again if is not moving
                OH_NO_YOURE_NOT_MOVING = min_block_diff < set.block_side_num and min_block_diff_dist <= 2
            # end if block diff
        # end for step, scrshot
        
        # calculate score: if (min_diff > thresold) then 1 else (min_diff / thresold)
        if min_full_diff_dist > 0 :
            full_score = min_full_diff / set.good_thrshld if min_full_diff < set.good_thrshld else 1.0
        if min_block_diff_dist > 0 :
            block_score = min_block_diff / set.block_num if min_block_diff < set.block_side_num * 2 else 1.0
        
        if full_score >= block_score :
            diff_score, diff_dist = full_score, min_full_diff_dist
        else :
            diff_score, diff_dist = block_score, min_block_diff_dist
        diff_score *= set.good_r
        #print("full diff:", min_full_diff, ", block diff:", min_block_diff)             
        
        if not OH_NO_YOURE_NOT_MOVING :
            #print("is moving\ndiff_score", diff_score)
            score = diff_score - set.bad_decline_rate * diff_dist
            return score if score > set.bad_r else set.bad_r
        else :
            #print("YOU SCREW")
            return set.bad_r
    