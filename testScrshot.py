from setting import *
# Countdown
countdown = 3
for i in range(countdown) :
    print(countdown - i)
    sleep(1.0)
scrshot = screenshot(region = GAME_REGION).resize((SCRSHOT_W, SCRSHOT_H), resample = Image.NEAREST)

noisy_scrshot = np.add((np.array(scrshot) / 255.5), np.random.uniform(low = -0.01, high = 0.01, size = (Q_INPUT_SHAPE)))
noisy_scrshot[noisy_scrshot > 1.0] = 1.0
noisy_scrshot[noisy_scrshot < 0.0] = 0.0
noisy_scrshot *= 255
noisy_scrshot = np.uint8(noisy_scrshot)
noisy_scrshot = Image.fromarray(noisy_scrshot, 'RGB')

scrshot.save("test_scrshot.png")
noisy_scrshot.save("test_noisy_scrshot.png")