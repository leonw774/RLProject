import numpy as np
from math import cos, sin, sqrt, pi
from PIL import Image
from pyautogui import click, screenshot
from time import sleep
from configure import get_game_region, Configuration as cfg
from directinputs import Keys

'''
Mouse movment parameter:
    - time (>= 0)
    - speed (> 0)
    - initial_angle (0~1 --> 0~2pi)
    - vertical_acc (-1~1 --> right~left)
    
- time:
    Indicates the duration of the movment in second
- speed
    Indicates the speed of the movment in pixels per second
- initial_angle
    Zero angle is pointing to the right.
    The angle spins counter-clickwise when the value increase to 1, where the angle is 2pi.
- vertical_acc
    The vertical acceleration that is applying to the movement.
    The speed value will not change becuase the acceleration is vertical to the speed.
    Negtive value means the force is pointing to the right side relative to the speed, while positive value means left side.
''' 

class Action:
    def __init__(self, parameterArray):
        self.time = parameterArray[0]
        self.speed = parameterArray[1] + 1e-6
        self.initial_angle = parameterArray[2] * 2 * pi
        self.vertical_acc = parameterArray[3]
        self.asarray = parameterArray
    
    def __str__(self):
        return str([self.time, self.speed, self.initial_angle, self.vertical_acc])

class GameAgent:

    GAME_REGION = get_game_region("Getting Over It")
    directInput = Keys()

    def add_scrshot_noise(self, noisy_scrshot):
        noisy_scrshot += np.random.uniform(low = -cfg.noise_range, high = cfg.noise_range, size = noisy_scrshot.shape)
        noisy_scrshot[noisy_scrshot > 1.0] = 1.0
        noisy_scrshot[noisy_scrshot < 0.0] = 0.0
        return noisy_scrshot
    # end def get_screen_rect
    
    def get_scrshot(self, wait_still = True, savefile = None) :
        # return screen-shot of game in np array in shape of cfg.shot_shape
        model_input_array = np.zeros(cfg.shot_shape)
        cur = screenshot(region = self.GAME_REGION).convert('RGB')
        i = 0
        while(wait_still and i <= cfg.shot_wait_max) :
            #print("waiting for no moving")
            sleep(cfg.shot_intv_time)
            pre = cur
            cur = screenshot(region = self.GAME_REGION).convert('RGB')
            if np.sum(np.array(cur)) <= 16 * self.GAME_REGION[2] * self.GAME_REGION[3] :
            # is black
                sleep(2)
                continue
            if np.sum(np.absolute((np.array(pre) - np.array(cur)) / 255.0)) < 16 * self.GAME_REGION[2] * self.GAME_REGION[3]:
            # not moving
                break
            i += 1
        
        if cfg.shot_c == 1 :
            cur = cur.convert('L')
            model_input_array = np.reshape(np.array(cur.resize(cfg.shot_resize, Image.BILINEAR)) / 255, cfg.shot_shape)
        elif cfg.shot_c == 3 :
            model_input_array[0] = np.array(cur.resize(cfg.shot_resize, Image.BILINEAR)) / 255
        else :
            raise Exception("shot_c isn't right.")
        if savefile : cur.save(savefile)
        return np.array(cur), model_input_array # oringal format, model input format
    # end def get_scrshot
    
    def control(self, action):
        #print(action)
        dx = cos(action.initial_angle)
        dy = sin(action.initial_angle)
        for _ in range(int(action.time * 1000)):
            self.directInput.directMouse(int(dx * action.speed), int(dy * action.speed))
            dx -= dy * action.vertical_acc
            dy += dx * action.vertical_acc
            len = sqrt(dx*dx + dy*dy)
            dx /= len
            dx /= len
            #sleep(cfg.control_intv_time)

    def newgame(self) :
        sleep(1)
        # click "NEW GAME"
        while(1) : # sometimes the game is not responsive to keybroad, you have to try more times
            shot1 = np.array(screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize))
            click(self.GAME_REGION[0] + self.GAME_REGION[2] * 0.70, self.GAME_REGION[1] + self.GAME_REGION[3] * 0.345)
            sleep(0.2)
            shot2 = np.array(screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize))
            if np.sum(np.abs(shot1 - shot2)) > cfg.diff_thrshld : break
            sleep(0.3)
        sleep(8)
    
    def quitgame(self) :
        sleep(1)
        # push ESC
        while(1) : # sometimes the game is not responsive to keybroad, you have to try more times
            shot1 = np.array(screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize))
            self.directInput.directKey("ESC")
            sleep(0.2)
            shot2 = np.array(screenshot(region = self.GAME_REGION).convert('RGB').resize(cfg.shot_resize))
            if np.sum(np.abs(shot1 - shot2)) > cfg.diff_thrshld : break
        # click "QUIT"
        click(self.GAME_REGION[0] + self.GAME_REGION[2] * 0.15, self.GAME_REGION[1] + self.GAME_REGION[3] * 1.05)
        sleep(12)
# end class GameAgent