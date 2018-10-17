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
block_h = set.compare_block_size[1]
block_w = set.compare_block_size[2]

print(set.block_diff_good_thrshld, set.block_score_good_thrshld)

#count_down
for i in range(10) :
    print(10 - i)
    sleep(1.0)

saved_scrshot_count = 0
array_scrshot = np.zeros(set.shot_shape)

while(True) :
    sleep(0.2)
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
    
    if np.sum(np.absolute(pre_array_scrshot - array_scrshot)) < set.no_move_thrshld : continue
    
    if len(scrshotList) == 0 :
        scrshotList.append(array_scrshot)
        print("map", saved_scrshot_count, "added")
        scrshot.save("map/" + str(saved_scrshot_count) + ".png")
        saved_scrshot_count += 1
    else :
        '''
        min_block_diff = 2147483648
        # blocks image diff
        # this algorithm is bigO(n^3), very slow. Only use if CPU is good
        for this_scrshot in scrshotList :
            # make compare_blocks
            # hack from stackoverflow :
            compare_blocks = array_scrshot.reshape(set.block_side_num, block_h, -1, block_w, set.shot_c).swapaxes(1,2).reshape(-1, block_h, block_w, set.shot_c)
            
            compare_result_array = np.full((set.block_num), 2147483648)
            for n in range(set.block_num) :
                i = 0
                j = 0
                while(i+block_h < set.shot_h) :
                    while(j+block_w < set.shot_w) :
                        compare_result_array[n] = min(
                            compare_result_array[n],
                            np.sum(np.absolute(
                                this_scrshot[i:i+block_h, j:j+block_w] - compare_blocks[n]
                            ))
                        )
                        j += set.compare_stride
                    i += set.compare_stride
            #print(compare_result_array)
            # compare_result :
            # 0 --> there is same (diff smaller then thresold)
            # 1 --> there is no same (diff larger)
            block_diff = np.sum((compare_result_array > set.block_diff_good_thrshld).astype(np.int))
            if block_diff < min_block_diff : min_block_diff = block_diff
        # end for this_scrshot
        print(min_block_diff)
        if min_block_diff != 2147483648 and min_block_diff > set.block_score_good_thrshld - 1 :
            scrshotList.append(array_scrshot)
            print("map", saved_scrshot_count, "added")
            scrshot.save("map/" + str(saved_scrshot_count) + ".png")
            saved_scrshot_count += 1
        # end if block diff
        '''
    
        min_diff = 2147483648
        for this_scrshot in scrshotList :
            d = np.sum(np.absolute(this_scrshot - array_scrshot))
            if d < min_diff : min_diff = d
        
        if min_diff > set.good_thrshld:
            scrshotList.append(array_scrshot)
            print("map", saved_scrshot_count, "added")
            scrshot.save("map/" + str(saved_scrshot_count) + ".png")
            saved_scrshot_count += 1
    
