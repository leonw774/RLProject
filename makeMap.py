'''
    HOW TO USE THIS
Once you run makeMap.py, it will monitor and screenshot the game.
The player have to play the game in the "right" way so that
AI could learn from it.  
'''
import numpy as np
from PIL import Image
from time import sleep
from setting import TrainingSetting as set
from pyautogui import click, screenshot
from directinputs import Keys
directKeys = Keys()

GameRegion = set.get_game_region("Getting Over It")
scrshotList = []

#count_down
for i in range(10) :
    print(10 - i)
    sleep(1.0)

saved_scrshot_count = 0
array_scrshot = np.zeros((int(GameRegion[3]), int(GameRegion[2]), 3))

while(True) :
    pre_array_scrshot = array_scrshot
    sleep(0.01)
    # get screen shot
    scrshot = (screenshot(region = GameRegion)).convert('RGB')
    array_scrshot = np.array(scrshot) / 255.5
    
    if np.sum(np.absolute(pre_array_scrshot - array_scrshot)) < 0.15 * GameRegion[2] * GameRegion[3]:    
        if len(scrshotList) == 0 :
            scrshotList.append(array_scrshot)
            print("map", saved_scrshot_count, "added")
            scrshot.save("map-new/" + str(saved_scrshot_count) + ".png")
            saved_scrshot_count += 1
        else :
            min_diff = 2147483648
            for this_scrshot in scrshotList :
                d = np.sum(np.absolute(this_scrshot - array_scrshot))
                if d < min_diff : min_diff = d
            
            if min_diff > 0.05 * GameRegion[2] * GameRegion[3]:
                scrshotList.append(array_scrshot)
                print("map", saved_scrshot_count, "added")
                scrshot.save("map-new/" + str(saved_scrshot_count) + ".png")
                saved_scrshot_count += 1
    
