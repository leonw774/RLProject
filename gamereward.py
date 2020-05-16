import os
import numpy as np
from PIL import Image
from configure import Configuration as cfg

def get_map_list():
    maplist = []
    for filename in cfg.map_name_list:
        if cfg.shot_c == 1:
            map = Image.open(cfg.map_path + filename).convert('L').resize(cfg.shot_resize, resample = Image.NEAREST)
            array_map = np.reshape(np.array(map) / 255.5, (cfg.shot_h, cfg.shot_w, 1))
        elif cfg.shot_c == 3:
            map = Image.open(cfg.map_path + filename).convert('RGB').resize(cfg.shot_resize, resample = Image.NEAREST)
            array_map = np.array(map) / 255.5
        maplist.append(array_map)
    return maplist
# end def get_maplist

MAP_LIST = get_map_list()
 
def getCurMap(scrshot):
    min_diff = 2147483648
    cur_map = -1
    for this_mapnum, this_mapshot in enumerate(MAP_LIST):
        d = np.sum(np.absolute(this_mapshot - scrshot))
        if d <= min_diff:
            min_diff = d
            if this_mapnum > cur_map : cur_map = this_mapnum
    return cur_map
# end def getCurMap

'''
class Reward():
    def __init__(self, base_reward, thresold):
        pass
    
    def getReward(self, pre_scrshot, cur_scrshot):
        pass
    
    def clear(self):
        pass
'''

class PixelDiffReward():
    def __init__(self, base_reward = cfg.base_reward, thresold = cfg.diff_thrshld):
        self.memorys = []
        self.base_reward = base_reward
        self.thresold = thresold
    
    def getReward(self, pre_scrshot, cur_scrshot):
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        if len(self.memorys) == 0:
            self.memorys.append(pre_scrshot)
        
        min_diff = 2147483648
        min_diff_index = -1

        for cur_mem_index, cur_memshot in enumerate(self.memorys):
            d = np.sum(np.absolute(cur_memshot - cur_scrshot))
            if d <= min_diff:
                min_diff = d
                if cur_mem_index > min_diff_index:
                    min_diff_index = cur_mem_index
        
        if (min_diff > self.thresold):
            self.memorys.append(cur_scrshot)
            return self.base_reward
        else:
            # -b * ((N - p) / N) * gamma
            return -self.base_reward * (len(self.memorys) - min_diff_index - 1) / len(self.memorys) * cfg.gamma
    # end def getReward
    
    def clear(self):
        self.memorys = []
# end class DiffReward

class TieredPixelDiffReward():

    class TierMemory():
        def __init__(self, scrshot, linkfrom) : 
            self.scrshot = scrshot
            self.linkfrom = linkfrom
            self.tier = linkfrom + 1
        
        def update(self, new_linkfrom):
            self.linkfrom = new_linkfrom
            self.tier = new_linkfrom + 1
    # end class TierMemory

    def __init__(self, base_reward = cfg.base_reward, thresold = cfg.diff_thrshld):
        self.memorys = []
        self.base_reward = base_reward
        self.thresold = thresold
        self.rev_gamma = (1 / cfg.gamma)
    
    def getReward(self, pre_scrshot, cur_scrshot):
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        if len(self.memorys) == 0:
            self.memorys.append(TierMemory(pre_scrshot, -1))
        
        min_cur_diff = 2147483648
        cur_pos = -1
        
        min_pre_diff = 2147483648
        pre_pos = -1

        for cur_mem_index, this_mem in enumerate(self.memorys):
            d = np.sum(np.absolute(this_mem.scrshot - cur_scrshot))
            if d <= min_cur_diff:
                min_cur_diff = d
                if cur_mem_index > cur_pos:
                    cur_pos = cur_mem_index
            
            d = np.sum(np.absolute(this_mem.scrshot - pre_scrshot))
            if d <= min_pre_diff:
                min_pre_diff = d
                if cur_mem_index > pre_pos:
                    pre_pos = cur_mem_index
        
        if min_cur_diff > self.thresold:
            self.memorys.append(TierMemory(cur_scrshot, self.memorys[pre_pos].tier))
            return self.base_reward
        
        # update when cur_tier > pre_tier + 1
        # or cur_tier + 1 < pre_tier
        if self.memorys[cur_pos].tier > self.memorys[pre_pos].tier + 1:
            #print("update: pos", cur_pos, "of tier", self.memorys[cur_pos].tier, "linkfrom", pre_pos, "of tier", self.memorys[pre_pos].tier)
            self.memorys[cur_pos].update(self.memorys[pre_pos].tier)

        if cur_pos == pre_pos:
            return 0
        else:
            # r = base * s * (rg ^ |s|)
            score = self.memorys[cur_pos].tier - self.memorys[pre_pos].tier
            return self.base_reward * score * (self.rev_gamma ** abs(score))
    # end def getReward
    
    def clear(self):
        self.memorys = []
# end class TieredDiffReward

class MapReward():
    def __init__(self, base_reward = cfg.base_reward, thresold = None):
        self.base_reward = base_reward
        self.rev_gamma = (1 / cfg.gamma) ** (1 / len(self.mapList))
        # so that the reward of last map, after times gamma, is exactly equal to total_r
        
        #print("no_move: ", cfg.diff_thrshld)
        #print("base_score:", self.base_score)
        #print("decline_rate:", self.decline_rate)
    
    def getReward(self, pre_scrshot, cur_scrshot):
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        min_pre_diff = 2147483648
        pre_map = -1

        min_cur_diff = 2147483648
        cur_map = -1
        
        for this_mapnum, this_mapshot in enumerate(self.mapList):
            d = np.sum(np.absolute(this_mapshot - pre_scrshot))
            if d <= min_pre_diff:
                min_pre_diff = d
                if this_mapnum > pre_map : pre_map = this_mapnum
            
            d = np.sum(np.absolute(this_mapshot - cur_scrshot))
            if d <= min_cur_diff:
                min_cur_diff = d
                if this_mapnum > cur_map : cur_map = this_mapnum
        
        if pre_map == cur_map:
            return 0
        else:
            # r = base * m * (rg ^ m)
            pre_score = self.base_reward * pre_map * (self.rev_gamma ** pre_map)
            cur_score = self.base_reward * cur_map * (self.rev_gamma ** pre_map)
            return cur_score - pre_score
    # end def calReward
    
    def clear(self):
        return None
# end class MapReward

class DissimilarityReward():
    def __init__(self, base_reward = cfg.base_reward, thresold = cfg.diff_thrshld):
        self.memorys = []
        self.base_reward = base_reward
        self.regions = (9, 9)
        self.reg_pixel_thresold = thresold / self.regions[0] / self.regions[1]
    
    def getReward(self, pre_scrshot, cur_scrshot):
        '''
        divide scrshot into rectangle regions
        "scan" cur_scrshot onto pre_scrshot
            check if we find a pair of similar region in this "scan", then stores it 
        return dissimilarity as 1 - number_of_similar_regions / number_of_regions
        '''
        pre_scrshot = np.squeeze(pre_scrshot)
        cur_scrshot = np.squeeze(cur_scrshot)
        reg_width = cfg.shot_w // self.regions[0]
        reg_height = cfg.shot_h // self.regions[1]

        similar_regions = set()
        image_data_bin = list(np.arange(256) / 255)
        
        for moving_offset_x in range(1-self.regions[0], self.regions[0]):
            for moving_offset_y in range(1-self.regions[1], self.regions[1]):
                for still_x in range(self.regions[0]):
                    for still_y in range(self.regions[1]):
                        moving_x = still_x - moving_offset_x
                        moving_y = still_y - moving_offset_y
                        if moving_x >= 0 and moving_x < self.regions[0] and moving_y >= 0 and moving_y < self.regions[1]:
                            #print((moving_offset_x, moving_offset_y, still_x, still_y))
                            s_bound_x = still_x*reg_width
                            s_bound_y = still_y*reg_height
                            s_region = pre_scrshot[s_bound_y:s_bound_y+reg_height, s_bound_x:s_bound_x+reg_width]
                            s_hist = np.histogram(s_region, bins=image_data_bin)[0]
                            
                            m_bound_x = moving_x*reg_width
                            m_bound_y = moving_y*reg_height
                            m_region = cur_scrshot[m_bound_y:m_bound_y+reg_height, m_bound_x:m_bound_x+reg_width]
                            m_hist = np.histogram(m_region, bins=image_data_bin)[0]
                            
                            pixel_diff = np.sum(np.absolute(s_region - m_region))
                            hist_diff = np.sum(np.absolute(m_hist - s_hist)) / (m_region.size * 2)
                            #print((moving_x, moving_y), diff)
                            if pixel_diff < self.reg_pixel_thresold and hist_diff < 0.33:
                                similar_regions.add((moving_x, moving_y))
        
        print("find", len(similar_regions), "similar regions")
        return self.base_reward * (1 - len(similar_regions) / self.regions[0] / self.regions[1])
    
    def clear(self):
        pass
    
# end class SimilarityReward