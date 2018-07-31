import numpy as np
import pyautogui
import time
import math
import random
import datetime

from directkeys import Keys
directKeys = Keys()

mouse_angle_devision = 36

time.sleep(3.0)

for n in range(120) :
    control_id = random.randint(0, mouse_angle_devision + 1)
    if (control_id < mouse_angle_devision) :
        angle = 2 * math.pi * control_id / mouse_angle_devision
        offset_x = math.ceil(math.cos(angle) * 12)
        offset_y = math.ceil(math.sin(angle) * 12)
        
        for i in range(10) :
            directKeys.directMouse(offset_x, offset_y)
            time.sleep(0.001)
    
    time.sleep(0.1)