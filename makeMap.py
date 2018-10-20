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
array_scrshot = np.zeros(set.shot_shape)

while(True) :
    sleep(0.1)
    pre_array_scrshot = array_scrshot
    # get screen shot
    if set.shot_c == 1 :
        scrshot = (screenshot(region = GameRegion)).convert('L').resize(set.shot_resize, resample = Image.NEAREST)
        array_scrshot = np.reshape(np.array(scrshot) / 255.5, set.shot_shape)
        array_scrshot = array_scrshot[0]
    elif set.shot_c == 3 :
        scrshot = (screenshot(region = GameRegion)).convert('RGB').resize(set.shot_resize, resample = Image.NEAREST)
        array_scrshot = np.array(scrshot) / 255.5
    else :
        raise Exception("shot_c isn't right.")
    
    if np.sum(np.absolute(pre_array_scrshot - array_scrshot)) < set.no_move_thrshld :    
        if len(scrshotList) == 0 :
            scrshotList.append(array_scrshot)
            print("map", saved_scrshot_count, "added")
            scrshot.save("map/" + str(saved_scrshot_count) + ".png")
            saved_scrshot_count += 1
        else :
            min_diff = 2147483648
            for this_scrshot in scrshotList :
                d = np.sum(np.absolute(this_scrshot - array_scrshot))
                if d < min_diff : min_diff = d
            
            if min_diff > set.good_thrshld:
                scrshotList.append(array_scrshot)
                print("map", saved_scrshot_count, "added")
                scrshot.save("map/" + str(saved_scrshot_count) + ".png")
                saved_scrshot_count += 1
    
