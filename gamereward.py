import numpy as np
from PIL import Image
from configure import Configuration as cfg

class DiffReward() :
    def __init__(self, base_reward = cfg.base_reward, thresold = cfg.no_move_thrshld) :
        self.memorys = []
        self.base_reward = base_reward
        self.thresold = thresold
    
    def getReward(self, pre_scrshot, cur_scrshot) :
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        if len(self.memorys) == 0 :
            self.memorys.append(pre_scrshot)
        
        min_diff = 2147483648
        min_diff_pos = -1

        for this_num, this_shot in enumerate(self.memorys) :
            d = np.sum(np.absolute(this_shot - cur_scrshot))
            if d <= min_diff :
                min_diff = d
                if this_num > min_diff_pos :
                    min_diff_pos = this_num
        
        if (min_diff > self.thresold) :
            self.memorys.append(cur_scrshot)
            return self.base_reward
        else :
            # -b * ((N - p) / N) * gamma
            return -self.base_reward * (len(self.memorys) - min_diff_pos - 1) / len(self.memorys) * cfg.gamma
    # end def getReward
    
    def clear(self) :
        self.memorys = []
# end class DiffReward

class Memory() :
    def __init__(self, scrshot, linkfrom) : 
        self.scrshot = scrshot
        self.linkfrom = linkfrom
        self.tier = linkfrom + 1
    
    def update(self, new_linkfrom) :
        self.linkfrom = new_linkfrom
        self.tier = new_linkfrom + 1
# end class Memory

class TieredDiffReward() :
    def __init__(self, base_reward = cfg.base_reward, thresold = cfg.no_move_thrshld) :
        self.memorys = []
        self.base_reward = base_reward
        self.thresold = thresold
        self.rev_gamma = (1 / cfg.gamma)
    
    def getReward(self, pre_scrshot, cur_scrshot) :
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        if len(self.memorys) == 0 :
            self.memorys.append(Memory(pre_scrshot, -1))
        
        min_cur_diff = 2147483648
        cur_pos = -1
        
        min_pre_diff = 2147483648
        pre_pos = -1

        for this_num, this_mem in enumerate(self.memorys) :
            d = np.sum(np.absolute(this_mem.scrshot - cur_scrshot))
            if d <= min_cur_diff :
                min_cur_diff = d
                if this_num > cur_pos :
                    cur_pos = this_num
            
            d = np.sum(np.absolute(this_mem.scrshot - pre_scrshot))
            if d <= min_pre_diff :
                min_pre_diff = d
                if this_num > pre_pos :
                    pre_pos = this_num
        
        if min_cur_diff > self.thresold :
            self.memorys.append(Memory(cur_scrshot, self.memorys[pre_pos].tier))
            return self.base_reward
        
        # update when cur_tier > pre_tier + 1
        # or cur_tier + 1 < pre_tier
        if self.memorys[cur_pos].tier > self.memorys[pre_pos].tier + 1 :
            #print("update: pos", cur_pos, "of tier", self.memorys[cur_pos].tier, "linkfrom", pre_pos, "of tier", self.memorys[pre_pos].tier)
            self.memorys[cur_pos].update(self.memorys[pre_pos].tier)

        if cur_pos == pre_pos :
            return 0
        else :
            # r = base * s * (rg ^ |s|)
            score = self.memorys[cur_pos].tier - self.memorys[pre_pos].tier
            return self.base_reward * score * (self.rev_gamma ** abs(score))
    # end def getReward
    
    def clear(self) :
        self.memorys = []
# end class TieredDiffReward

class MapReward() :
    def __init__(self, base_reward = cfg.base_reward) :
        self.mapList = []
        for filename in cfg.mapname_list :
            if cfg.shot_c == 1 :
                map = Image.open("map/" + filename).convert('L').resize(cfg.shot_resize, resample = Image.NEAREST)
                array_map = np.reshape(np.array(map) / 255.5, (cfg.shot_h, cfg.shot_w, 1))
            elif cfg.shot_c == 3 :
                map = Image.open("map/" + filename).convert('RGB').resize(cfg.shot_resize, resample = Image.NEAREST)
                array_map = np.array(map) / 255.5
            self.mapList.append(array_map)
        
        self.base_reward = base_reward
        self.rev_gamma = (1 / cfg.gamma) ** (1 / len(self.mapList))
        # so that the reward of last map, after times gamma, is exactly equal to total_r
        
        #print("no_move: ", cfg.no_move_thrshld)
        #print("base_score:", self.base_score)
        #print("decline_rate:", self.decline_rate)
    
    def getReward(self, pre_scrshot, cur_scrshot) :
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
        
        if pre_map == cur_map :
            return 0
        else :
            # r = base * m * (rg ^ m)
            pre_score = self.base_reward * pre_map * (self.rev_gamma ** pre_map)
            cur_score = self.base_reward * cur_map * (self.rev_gamma ** pre_map)
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
 