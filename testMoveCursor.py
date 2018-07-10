import sys
import time
import pyautogui

cursor_pos_description = ["up", "left", "down", "right"]
screen_w, screen_h = pyautogui.size();
'''
for i in range(10) :
    m = i % 4
    print(cursor_pos_description[m])
    if m == 0 :
        pyautogui.moveTo(screen_w / 2, 100) # up
    elif m == 1 :
        pyautogui.moveTo(100, screen_h / 2) # left
    elif m == 2 :
        pyautogui.moveTo(screen_w / 2, screen_h - 100) # down 
    elif m == 3 :
        pyautogui.moveTo(screen_w - 100, screen_h / 2) # right
    time.sleep(1)
'''
for i in range(10) :
    print(pyautogui.position())
    time.sleep(1)
