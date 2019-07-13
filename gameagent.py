import numpy as np
import math
from PIL import Image
from pyautogui import click, screenshot
from time import sleep
from configure import get_game_region, Configuration as cfg
from directinputs import Keys

class GameAgent :
    
    GAME_REGION = get_game_region("Getting Over It")
    directInput = Keys()
    
    def getScreenshot(self, wait_still = True, savefile = None) :
        # return screen-shot of game in np array in shape of cfg.shot_shape
        array_scrshot = np.zeros(cfg.shot_shape)
        cur = screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize)
        i = 0
        while(wait_still and i <= cfg.shot_wait_max) :
            #print("waiting for no moving")
            sleep(cfg.shot_intv_time)
            pre = cur
            cur = screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize)
            if np.sum(np.array(cur)) <= 256 : # is black
                sleep(1.6)
                continue
            if np.sum(np.absolute((np.array(pre) - np.array(cur)) / 256.0)) < 33 * cfg.no_move_thrshld :
                break
            i += 1
        
        if cfg.shot_c == 1 :
            array_scrshot = np.reshape(np.array(cur.convert('L')) / 255.5, cfg.shot_shape)
        elif cfg.shot_c == 3 :
            array_scrshot[0] = np.array(cur) / 255.5
        else :
            raise Exception("shot_c isn't right.")
        if savefile : cur.save(savefile)
        return array_scrshot
    # end def getScreenshot
    
    def doControl(self, id) :
        intv_time = 0.001
         # is straight
        if id < cfg.mouse_straight_angles * 2 :
        
            if id < cfg.mouse_straight_angles :
                # slow
                delta, distance = 3, 2400
            else :
                # fast
                delta, distance = 20, 3600
            
            angle = 2 * math.pi * id / cfg.mouse_straight_angles
            d_x = math.ceil(math.cos(angle) * delta)
            d_y = math.ceil(math.sin(angle) * delta)
            
            for i in range(distance // delta) :
                self.directInput.directMouse(d_x, d_y)
                sleep(intv_time)
            if id >= cfg.mouse_straight_angles :
                sleep(0.02)
        # is round
        else :
            id -= cfg.mouse_straight_angles * 2
            
            if id < cfg.mouse_round_angles * 2 :
                is_clockwise = 1
            else :
                is_clockwise = -1
                id -= cfg.mouse_round_angles * 2
            
            if id < cfg.mouse_round_angles :
                # slow
                radius, delta, proportion = 1000, 4, 0.8
            else :
                # fast
                radius, delta, proportion = 1500, 16, 0.8
            
            angle_divide = 36.0
            angle_bias = 4.0
            angle_offset = (id / cfg.mouse_round_angles) + angle_bias / angle_divide
            edge_leng = math.floor(2 * radius * math.sin(math.pi / angle_divide))
            # we cut a circle's edge into circular arcs.
            # each arcs is similar to the base of an isosceles triangle
            # an isosceles triangle with legs = r and apex = a, so it base = 2r * sin(a/2)
            
            for i in range(int(angle_divide * proportion)) : 
                angle = 2 * math.pi * (i * is_clockwise / angle_divide + angle_offset)
                d_x = math.ceil(math.cos(angle) * delta)
                d_y = math.ceil(math.sin(angle) * delta)
                for j in range(edge_leng // delta) :
                    self.directInput.directMouse(d_x, d_y)
                    sleep(intv_time)
            sleep(0.01)
        sleep(cfg.control_pause)
    # end def doControl()
    
    def newgame(self) :
        sleep(1)
        # click "NEW GAME"
        while(1) : # sometimes the game is not responsive to keybroad, you have to try more times
            shot1 = np.array(screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize))
            click(self.GAME_REGION[0] + self.GAME_REGION[2] * 0.70, self.GAME_REGION[1] + self.GAME_REGION[3] * 0.35)
            sleep(0.2)
            shot2 = np.array(screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize))
            if np.sum(np.abs(shot1 - shot2)) > cfg.no_move_thrshld : break
            sleep(0.3)
        sleep(7)
    
    def quitgame(self) :
        sleep(1)
        # push ESC
        while(1) : # sometimes the game is not responsive to keybroad, you have to try more times
            shot1 = np.array(screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize))
            self.directInput.directKey("ESC")
            sleep(0.2)
            shot2 = np.array(screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize))
            if np.sum(np.abs(shot1 - shot2)) > cfg.no_move_thrshld : break
        # click "QUIT"
        click(self.GAME_REGION[0] + self.GAME_REGION[2] * 0.15, self.GAME_REGION[1] + self.GAME_REGION[3] * 1.05)
        sleep(12)
# end class GameAgent