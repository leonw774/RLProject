from setting import *
# Countdown
countdown = 3
for i in range(countdown) :
    print(countdown - i)
    sleep(1.0)
scrshot = (screenshot(region = GAME_REGION)).resize((SCRSHOT_W, SCRSHOT_H), resample = 0)
scrshot.save("testscrshot.png")