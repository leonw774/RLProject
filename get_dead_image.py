import numpy as np
from PIL import ImageChops, Image
import pyautogui

screen_w, screen_h = pyautogui.size()
isdead_region = (screen_w / 2 - 20, screen_h / 3 - 20, 40, 40)
isdead = pyautogui.screenshot(region = isdead_region)
isdead.save("isdead_image.png")

    

