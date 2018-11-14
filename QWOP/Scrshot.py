#from setting import *
import pyautogui
import msvcrt
# Countdown
countdown = 425
#for i in range(countdown) :
 #   print(countdown - i)
  #  sleep(1.0)
GAME_REGION = 490, 320 , 910 , 120
path = r"C:\Users\lin2\Documents\GitHub\QWOPProject_now\QWOP\dead.png"
print(countdown)
while countdown < 428:
    char = msvcrt.getch()
    if char == b'a':
        countdown = countdown + 1
        print(countdown)
        scrshot = pyautogui.screenshot(region = GAME_REGION)
        #path2 = path + str(countdown) + ".png"
        scrshot.save(path)





