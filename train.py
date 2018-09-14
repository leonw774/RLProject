import sys
import random
import math
import numpy as np
from PIL import Image
from time import sleep
from datetime import datetime, timedelta
from pyautogui import click, screenshot
from win32 import win32gui
from keras.models import Model

from setting import TrainingSetting as set
from stepqueue import StepQueue
from directinputs import Keys
from QNet import QNet

class Train() :
    
    def __init__(self, use_weight_file = None) :
        
        self.GameRegion = self.get_game_region("Getting Over It")
        
        self.directInput = Keys()
        self.Q = QNet(set.model_input_shape, set.actions_num)
        self.Q.summary()
        self.Q.compile(loss = "mse", optimizer = set.model_optimizer)
        
        if use_weight_file :
            self.Q.load_weights(use_weight_file)
        
        if set.steps_update_target > 0 :
            self.Q_target = QNet(set.model_input_shape, set.actions_num)
            self.Q_target.set_weights(self.Q.get_weights())
            
        self.A = []
        for i in range(set.actions_num) :
            action_onehot = np.zeros((1, set.actions_num))
            action_onehot[0, i] = 1
            self.A.append(action_onehot)
    
    def get_game_region(self, title) :
        if title :
            gamewin = win32gui.FindWindow(None, title)
            if not gamewin:
                raise Exception('window title not found')
            #get the bounding box of the window
            x1, y1, x2, y2 = win32gui.GetWindowRect(gamewin)
            
            y1 += 50 # get rid of window bar
            y2 -= 10
            x1 += 10
            x2 -= 10
            
            return (x1, y1, (x2 - x1 + 1), (y2 - y1 + 1))
        else :
            raise Exception("no window title was given.")
    # end get_game_region
    
    def count_down(self, cd) :
        for i in range(cd) :
            print(cd - i)
            sleep(1.0)
            
    def get_screenshot(self, num = 1, savefile = None) :
        sleep(set.scrshot_intv_time)
        array_scrshot = np.zeros(set.scrshot_shape)
        while(True) :
            if set.color_size == 1 :
                scrshot = (screenshot(region = self.GameRegion)).convert('L').resize(set.scrshot_resize, resample = Image.NEAREST)
                array_scrshot = np.reshape(np.array(scrshot) / 255.5, set.scrshot_shape)
            elif set.color_size == 3 :
                scrshot = (screenshot(region = self.GameRegion)).convert('RGB').resize(set.scrshot_resize, resample = Image.NEAREST)
                array_scrshot[0] = np.array(scrshot) / 255.5
            else :
                raise Exception("color_size isn't right.")
            if savefile : scrshot.save(savefile)
            if np.sum(array_scrshot) < 0.001 :
                sleep(1.0)
            else :
                return array_scrshot
    # end def get_screenshot
    
    def add_noise(self, noisy_scrshot) :
        noisy_scrshot += np.random.uniform(low = -set.noise_range, high = set.noise_range, size = noisy_scrshot.shape)
        noisy_scrshot[noisy_scrshot > 1.0] = 1.0
        noisy_scrshot[noisy_scrshot < 0.0] = 0.0
        return noisy_scrshot
    # end def get_screen_rect
    
    def do_control(self, id) : 
        '''
        相對方向的滑動
        [slow moving x angle_num, fast moving x angle_num]
        '''
        slow_distance = 2400 # pixels
        fast_distance = 4000 # pixels
        slow_intv_distance = 5 # pixels
        fast_intv_distance = 30
        intv_time = 0.0025
        
        
        if id < set.mouse_angle_devision :
            intv_distance, distance = slow_intv_distance, slow_distance
        else :
            intv_distance, distance = fast_intv_distance, fast_distance
        
        angle = 2 * math.pi * id / set.mouse_angle_devision
        offset_x = math.ceil(math.cos(angle) * intv_distance)
        offset_y = math.ceil(math.sin(angle) * intv_distance)
        
        for i in range(distance // intv_distance) :
            self.directInput.directMouse(offset_x, offset_y)
            sleep(intv_time)
        sleep(set.do_control_pause)
    # end def do_control()
    
    def click_newgame(self) :
        # click "NEW GAME"
        click(self.GameRegion[0] + self.GameRegion[2] * 0.66, self.GameRegion[1] + self.GameRegion[3] * 0.36)
        sleep(7)
    
    def click_quitgame(self) :
        sleep(0.2)
        # push ESC
        self.directInput.directKey("ESC")
        sleep(0.2)
        # click "QUIT"
        click(self.GameRegion[0] + self.GameRegion[2] * 0.21, self.GameRegion[1] + self.GameRegion[3] * 0.95)
        sleep(7)
    
    def fit(self) :
        '''
        We will train Q, at time t, as:
            y_pred = Q([state_t, a_t])
            y_true = r_t + gamma * max(Q_target([state_t, for a in A]))
        update Q's weight in mse
        after a number of steps, copy Q to Q_target.
        '''
        
        for e in range(set.epoches) :
            
            self.click_newgame()
            
            stepQueue = StepQueue()
            total_reward = 0
            cur_scrshots = np.zeros((1, set.scrshot_h, set.scrshot_w, set.color_size * set.scrshot_n))

            for n in range(set.steps_epoch) :
                cur_shot = self.get_screenshot()
                cur_scrshots[:,:,:, : -set.color_size] = cur_scrshots[:,:,:, set.color_size: ] # dequeue
                cur_scrshots[:,:,:, -set.color_size : ] = cur_shot # enqueue

                # make action
                if n <= set.scrshot_n or random.random() < max(set.eps_min, set.epsilon * (set.eps_decay ** e)) :
                    cur_action = random.randrange(set.actions_num)
                else :
                    cur_action = np.argmax(self.Q.predict(self.add_noise(cur_scrshots)))
                
                self.do_control(cur_action)
                
                nxt_shot = self.get_screenshot()
                cur_reward = stepQueue.calReward(cur_shot, nxt_shot) # pre-action, after-action
                #print(cur_action, ",", cur_reward)
                if cur_reward == "stuck" :
                    print("at step", n)
                    screenshot(region = self.GameRegion).save("stuck_at_epoch" + str(e) + ".png")
                    break
                
                total_reward += cur_reward
                stepQueue.addStep(cur_shot, cur_action, cur_reward, nxt_shot)
                
                if stepQueue.getLength() > set.train_size and n > set.train_thrshld and n % set.steps_train == 0 :
                    # Experience Replay
                    random_step = random.randint(1, stepQueue.getLength() - set.train_size)
                    train_cur_scrshots, train_actions, train_rewards, train_nxt_shots = stepQueue.getStepsAsArray(random_step, set.train_size)
                    train_input_scrshots = np.zeros((set.train_size, set.scrshot_h, set.scrshot_w, set.color_size * set.scrshot_n))
                    
                    # make replay reward array
                    replay_rewards = np.zeros((set.train_size, set.actions_num))
                    for j in range(set.train_size) :
                        replay_rewards[j, int(train_actions[j])] = train_rewards[j]
                    
                    # make next predicted reward array and train input array at same time
                    nxt_rewards = np.zeros((set.train_size, set.actions_num))
                    for j in range(set.train_size) :
                        train_input_scrshots[j,:,:, : -set.color_size] = stepQueue.getScrshotAsArray(random_step + j - set.scrshot_n, set.scrshot_n - 1)
                        train_input_scrshots[j,:,:, -set.color_size : ] = train_cur_scrshots[j]
                        train_input_scrshots[j] = self.add_noise(train_input_scrshots[j])
                        this_predict_reward_input = train_input_scrshots[j].reshape((1, set.scrshot_h, set.scrshot_w, set.scrshot_n*set.color_size))
                        if set.steps_update_target > 0 :
                            nxt_rewards[j] = self.Q.predict(this_predict_reward_input)
                        else :
                            nxt_rewards[j] = self.Q_target.predict(this_predict_reward_input)
                        
                    train_targets_rewards = replay_rewards + (nxt_rewards * set.gamma)
                    
                    #print("replay_rewards\n", replay_rewards[0])
                    #print("nxt_rewards\n", nxt_rewards[0])
                    #print("train_targets\n", train_targets[0])
                    
                    loss = self.Q.train_on_batch(train_input_scrshots, train_targets_rewards)
                    #print("loss: ", loss)
                    
                if set.steps_update_target > 0 and n % set.steps_update_target == 0 and n > set.train_thrshld :
                    #print("assign Qtarget")
                    self.Q_target.set_weights(self.Q.get_weights())
                    self.Q_target.save_weights("Q_target_weight.h5")
                    
                # end for(STEP_PER_EPOCH)  
            print("end epoch", e, "total_reward:", total_reward)
            stepQueue.clear()
            
            # Restart Game...
            self.click_quitgame()
            
        # end for(epoches)
        
        self.Q_target.save_weights("Q_target_weight.h5")
        
    # end def fit
    
    def eval(self, model_weight_name) :
        print("eval begin for:", model_weight_name)
        self.Q.load_weights(model_weight_name)
        
        # click "NEW GAME"
        self.click_newgame()
        
        stepQueue = StepQueue()
        total_reward = 0
        cur_scrshots = np.zeros((1, set.scrshot_h, set.scrshot_w, set.color_size * set.scrshot_n))
        for n in range(set.steps_test) :
            cur_shot = self.get_screenshot()
            cur_scrshots[:,:,:, : -set.color_size] = cur_scrshots[:,:,:, set.color_size: ] # dequeue
            cur_scrshots[:,:,:, -set.color_size : ] = cur_shot # enqueue
            
            if n <= set.scrshot_n or random.random() < set.eps_test :
                cur_action = random.randrange(set.actions_num)
            else :
                cur_action = np.argmax(self.Q.predict([self.add_noise(pre_shot), self.add_noise(cur_shot)]))
            
            self.click_newgame()
                
            self.do_control(cur_action)
            nxt_shot = self.get_screenshot()
            cur_reward = stepQueue.calReward(cur_shot, nxt_shot) # pre, cur
            if cur_reward == "stuck" : break
            stepQueue.addStep(cur_shot, cur_action, cur_reward, nxt_shot)
            total_reward += cur_reward
        
        del stepQueue
        print("eval end, totalReward:", totalReward)
        screenshot(region = self.GameRegion).save("eval_scrshot.png")
        
        # Exit Game...
        self.click_quitgame()

    # end def eval
    
    def random_action(self, times) :
        # click "NEW GAME"
        self.click_newgame()
        stepQueue = stepQueue()
        total_reward = 0
        for n in range(times) :
            cur_shot = self.get_screenshot()
            cur_action = random.randrange(set.actions_num)
            self.do_control(cur_action)
            nxt_shot = self.get_screenshot()
            cur_reward = stepQueue.calReward(cur_shot, nxt_shot)
            #print(cur_action, ",", cur_reward)
            if cur_reward == "stuck" :
                print("at step", n)
                screenshot(region = self.GameRegion).save("stuck_at_random.png")
                break
            stepQueue.addStep(cur_shot, cur_action, cur_reward, nxt_shot)
            total_reward += cur_reward
        
        del stepQueue
        print("eval end, totalReward:", total_reward)
        # Exit Game...
        self.click_quitgame()
    # end def
    
# end class Train

if __name__ == '__main__' :
    train = Train()
    train.count_down(3)
    starttime = datetime.now()
    #train.random_action(2500)
    train.fit()
    print(datetime.now() - starttime)
    #train.eval("Q_target_weight.h5")
    

