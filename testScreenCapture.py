import numpy as np
from PIL import ImageChops, Image
import pyautogui

screen_w, screen_h = pyautogui.size()
dead_image = Image.open("dead_screenshot.png")
isdead_region = (screen_w / 2 - 20, screen_h / 3 - 20, 40, 40)

for i in range(1) :
    #game_region = (50, 50, screen_w - 100, screen_h - 100)
    #capture = pyautogui.screenshot(region = game_region)
    
    isdead = pyautogui.screenshot(region = isdead_region)
    #isdead.save("dead_screenshot.png")
    bbox = ImageChops.difference(dead_image, isdead).getbbox()
    if bbox == None :
        print("is dead")
        break
    

