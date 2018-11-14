import numpy as np
from PIL import ImageChops, Image
import pyautogui

screen_w, screen_h = pyautogui.size()
isdead_region = 715, 85 , 170 ,85
isdead = pyautogui.screenshot(region = isdead_region)
scrshot = np.array(isdead)

im_frame = Image.open(r'C:\Users\lin2\Documents\GitHub\QWOPProject_now\QWOP\test_scrshot1.png')
np_frame = np.array(im_frame.getdata()).reshape(85,170,3)
    
result_arrray = scrshot- np_frame
result = np.sum(result_arrray)
if abs(result) < 0.1:
	print("True")
else:
	print("False")

(490, 320)
(1400, 320)
(1400, 440)
(490, 440)