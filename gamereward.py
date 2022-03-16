import os
import numpy as np
from math import sqrt
from PIL import Image
from skimage.feature import (match_descriptors, corner_peaks, corner_harris, BRIEF)
from skimage.color import rgb2gray
from configure import Configuration as cfg

'''
class Reward():
    def __init__(self, base_reward, thresold):
        pass
    
    def get_reward(self, pre_scrshot, cur_scrshot):
        pass
    
    def clear(self):
        pass
'''

class PixelDiffReward():
    def __init__(self):
        self.memeories = []
        self.base_reward = cfg.base_reward
        self.thresold = cfg.diff_thrshld
    
    def get_reward(self, pre_scrshot, cur_scrshot):
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        if len(self.memeories) == 0:
            self.memeories.append(pre_scrshot)
        
        min_diff = 1e10
        min_diff_index = -1

        for cur_mem_index, cur_memshot in enumerate(self.memeories):
            d = np.sum(np.absolute(cur_memshot - cur_scrshot))
            if d <= min_diff:
                min_diff = d
                if cur_mem_index > min_diff_index:
                    min_diff_index = cur_mem_index
        
        if (min_diff > self.thresold):
            self.memeories.append(cur_scrshot)
            return self.base_reward
        else:
            # -b * ((N - p) / N) * gamma
            return -self.base_reward * (len(self.memeories) - min_diff_index - 1) / len(self.memeories) * cfg.gamma
    # end def get_reward
    
    def clear(self):
        self.memeories = []
# end class DiffReward

""" 
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

    def __init__(self):
        self.memeories = []
        self.base_reward = base_reward
        self.thresold = thresold
        self.rev_gamma = (1 / cfg.gamma)
    
    def get_reward(self, pre_scrshot, cur_scrshot):
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        if len(self.memeories) == 0:
            self.memeories.append(self.TierMemory(pre_scrshot, -1))
        
        min_cur_diff = 1e10
        cur_pos = -1
        
        min_pre_diff = 1e10
        pre_pos = -1

        for cur_mem_index, this_mem in enumerate(self.memeories):
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
            self.memeories.append(self.TierMemory(cur_scrshot, self.memeories[pre_pos].tier))
            return self.base_reward
        
        # update when cur_tier > pre_tier + 1
        # or cur_tier + 1 < pre_tier
        if self.memeories[cur_pos].tier > self.memeories[pre_pos].tier + 1:
            #print("update: pos", cur_pos, "of tier", self.memeories[cur_pos].tier, "linkfrom", pre_pos, "of tier", self.memeories[pre_pos].tier)
            self.memeories[cur_pos].update(self.memeories[pre_pos].tier)

        if cur_pos == pre_pos:
            return 0
        else:
            # r = base * s * (rg ^ |s|)
            score = self.memeories[cur_pos].tier - self.memeories[pre_pos].tier
            return self.base_reward * score * (self.rev_gamma ** abs(score))
    # end def get_reward
    
    def clear(self):
        self.memeories = []
# end class TieredDiffReward """

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

def get_cur_map(scrshot):
    min_diff = 2147483648
    cur_map = -1
    for this_mapnum, this_mapshot in enumerate(MAP_LIST):
        d = np.sum(np.absolute(this_mapshot - scrshot))
        if d <= min_diff:
            min_diff = d
            if this_mapnum > cur_map : cur_map = this_mapnum
    return cur_map
# end def get_cur_map

MAP_LIST = get_map_list()
    
class MapReward():
    def __init__(self):
        self.base_reward = cfg.base_reward
        # make the reward of last map, after times gamma, exactly equals to total_r
        self.rev_gamma = (1 / cfg.gamma) ** (1 / len(MAP_LIST))
        #print("no_move: ", cfg.diff_thrshld)
        #print("base_score:", self.base_score)
        #print("decline_rate:", self.decline_rate)
    # end __init__
        
    def get_reward(self, pre_scrshot, cur_scrshot):
        pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action
        
        min_pre_diff = 1e10
        pre_map = -1

        min_cur_diff = 1e10
        cur_map = -1
        
        for this_mapnum, this_mapshot in enumerate(MAP_LIST):
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

class FeatureDisplacementReward():
    def __init__(self):
        self.memeories = []
        self.fdMax = cfg.shot_h * cfg.shot_h + cfg.shot_w * cfg.shot_w
        self.extractor = BRIEF()
    
    def median_outlier_filter(self, signal, threshold=3):
        signal = signal.copy()
        diff = np.abs(signal - np.median(signal))
        median_diff = np.median(diff)
        s = diff / (float(median_diff) + 1e-6)
        mask = s > threshold
        signal[mask] = np.median(signal)
        return signal
    
    def get_feature_displacment(self, img1, img2):

        # if cfg.shot_c == 3:
        #     img1 = rgb2gray(img1)
        #     img2 = rgb2gray(img2)
        keypoints1 = corner_peaks(corner_harris(img1), min_distance=5, threshold_rel=0.02)
        keypoints2 = corner_peaks(corner_harris(img2), min_distance=5, threshold_rel=0.02)

        self.extractor.extract(img1, keypoints1)
        keypoints1 = keypoints1[self.extractor.mask]
        descriptors1 = self.extractor.descriptors

        self.extractor.extract(img2, keypoints2)
        keypoints2 = keypoints2[self.extractor.mask]
        descriptors2 = self.extractor.descriptors
        
        if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
            return None # no feature found

        matches = match_descriptors(descriptors1, descriptors2, cross_check=True, max_ratio=0.85)
        if matches.shape[0] < 4: return None
        dist = np.sum((keypoints1[matches[:,0]] - keypoints2[matches[:,1]]) ** 2, axis = 1)
        return np.mean(self.median_outlier_filter(dist))

    def get_reward(self, pre_scrshot, cur_scrshot):
        # pre_scrshot = np.squeeze(pre_scrshot) # before action
        cur_scrshot = np.squeeze(cur_scrshot) # after action

        # pre_scrshot = (pre_scrshot*255).astype(dtype=int)
        cur_scrshot = (cur_scrshot*255).astype(dtype=int)

        min_mem_fd = self.fdMax
        min_mem_id = -1
        for id, mem_shot in enumerate(self.memeories):
            fd = self.get_feature_displacment(cur_scrshot, mem_shot)
            if fd:
                min_mem_fd = min(fd, min_mem_fd)
                min_mem_id = id
        if min_mem_id == len(self.memeories) - 1: # if cur_shot is closer to last memory position
            self.memeories.append(cur_scrshot)
            return cfg.base_reward * min_mem_fd / self.fdMax
        elif min_mem_id > 0: # cur_shot is closer to older memory position
            return 0
        else: # cur_scrshot & cur_scrshot have no similar features to memories at all
            self.memeories.append(cur_scrshot)
            cfg.base_reward


    def clear(self):
        self.memeories = []
    
# end class SimilarityReward