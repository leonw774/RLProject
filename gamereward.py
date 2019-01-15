import numpy as np
from PIL import Image
from setting import Setting as set

class MapReward :
    def __init__(self, base_reward = set.base_reward) :
        self.mapList = []
        for filename in set.mapname_list :
            if set.shot_c == 1 :
                map = Image.open("map/" + filename).convert('L').resize(set.shot_resize, resample = Image.NEAREST)
                array_map = np.reshape(np.array(map) / 255.5, (set.shot_h, set.shot_w, 1))
            elif set.shot_c == 3 :
                map = Image.open("map/" + filename).convert('RGB').resize(set.shot_resize, resample = Image.NEAREST)
                array_map = np.array(map) / 255.5
            self.mapList.append(array_map)
        
        self.base_score = base_reward
        self.decline_rate = (1 / set.gamma) ** (1 / len(self.mapList))
        # so that the reward of last map, after times gamma, is exactly equal to total_r
        
        #print("no_move: ", set.no_move_thrshld)
        #print("base_score:", self.base_score)
        #print("decline_rate:", self.decline_rate)
    
    def calReward(self, pre_scrshot, cur_scrshot) :
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        min_pre_diff = 2147483648
        pre_map = -1

        min_cur_diff = 2147483648
        cur_map = -1
        
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
        
        #if min_cur_diff >= set.no_move_thrshld * 3 :
            # not in the map!?
        #    return 0.0
        
        if pre_map == cur_map :
            return 0
        else :
            # r = base * m * (d ^ m)
            pre_score = self.base_score * pre_map * (self.decline_rate ** pre_map)
            cur_score = self.base_score * cur_map * (self.decline_rate ** pre_map)
            return cur_score - pre_score
    # end def calReward 
    
    def getCurMap(self, scrshot) :
        min_diff = 2147483648
        cur_map = -1
        for this_mapnum, this_mapshot in enumerate(self.mapList) :
            d = np.sum(np.absolute(this_mapshot - scrshot))
            if d <= min_diff :
                min_diff = d
                if this_mapnum > cur_map : cur_map = this_mapnum
        return cur_map
    # end def getCurMap
# end class MapReward
 
class DiffReward :
    def __init__(self, base_score = set.base_reward) :
        self.memoryList = []
        self.base_score = base_score
        self.thresold = set.no_move_thrshld
    
    def clear(self) :
        self.memoryList = []
    
    def calReward(self, pre_scrshot, cur_scrshot) :
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        if len(self.memoryList) == 0 :
            self.memoryList.append(pre_scrshot)
        
        min_diff = 2147483648
        min_diff_pos = -1

        for this_num, this_shot in enumerate(self.memoryList) :
            d = np.sum(np.absolute(this_shot - cur_scrshot))
            if d <= min_diff :
                min_diff = d
                if this_num > min_diff_pos :
                    min_diff_pos = this_num
        
        if (min_diff > self.thresold) :
            self.memoryList.append(cur_scrshot)
            return self.base_score
        else :
            return -self.base_score * (len(self.memoryList) - min_diff_pos - 1) * set.gamma #/ len(self.memoryList)
        
    # end def calReward
# end class MapReward