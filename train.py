import random
import math
import numpy as np
from PIL import Image
from time import sleep
from pyautogui import click, screenshot
from keras.models import Model

from setting import TrainingSetting as set
from setting import game_region
from statequeue import StateQueue
from directinputs import Keys
from QNet import QNet

class Train() :
    
    def __init__(self) :
        self.directInput = Keys()
        self.Q = QNet(set.model_input_shape, set.actions_num)
        self.Q.compile(loss = "mse", optimizer = set.model_optimizer)
        self.Q_target = QNet(set.model_input_shape, set.actions_num)
        self.Q_target.set_weights(self.Q.get_weights())
        self.A = []
        for i in range(set.actions_num) :
            action_onehot = np.zeros((1, set.actions_num))
            action_onehot[0, i] = 1
            self.A.append(action_onehot)
    
    def count_down(self, cd) :
        for i in range(cd) :
            print(cd - i)
            sleep(1.0)
            
    def get_screenshot(self) :
        sleep(set.scrshot_intv_time)
        array_scrshot = np.zeros(set.scrshot_shape)
        if set.color_size == 1 :
            scrshot = (screenshot(region = game_region)).convert('L').resize(set.scrshot_resize, resample = Image.NEAREST)
        elif set.color_size == 3 :
            scrshot = (screenshot(region = game_region)).convert('RGB').resize(set.scrshot_resize, resample = Image.NEAREST)
        else :
            raise Exception("color_size isn't right.")
        array_scrshot[0] = np.array(scrshot) / 255.5
        return array_scrshot
    # end def get_screenshot
    
    def add_noise(self, noisy_scrshot) :
        noisy_scrshot += np.random.uniform(low = -0.01, high = 0.01, size = set.scrshot_shape)
        noisy_scrshot[noisy_scrshot > 1.0] = 1.0
        noisy_scrshot[noisy_scrshot < 0.0] = 0.0
        return noisy_scrshot
    # end def get_screen_rect
    
    def do_control(self, id) : 
        '''
        相對方向的滑動
        未來或許可以嘗試在ActionSet中加入控制快慢和距離的選擇
        [slow moving x angle_num, fast moving x angle_num]
        '''
        slow_distance = 2000 # pixels
        fast_distance = 800 # pixels
        slow_intv = 8 # pixels
        fast_intv = 20
        intv_time = 0.0036
        
        if id < set.mouse_angle_devision :
            intv, distance = slow_intv, slow_distance
        else :
            intv, distance = fast_intv, fast_distance
        
        angle = 2 * math.pi * id / set.mouse_angle_devision
        offset_x = math.ceil(math.cos(angle) * intv)
        offset_y = math.ceil(math.sin(angle) * intv)
        
        for i in range(distance // intv) :
            self.directInput.directMouse(offset_x, offset_y)
            sleep(intv_time)
        sleep(set.do_control_pause)
    # end def do_control()
    
    def run(self) :
        '''
        We would train Q, at time t, as:
            y_pred = Q([state_t, a_t])
            y_true = r_t + gamma * max(Q_target([state_t, for a in A]))
        update Q's weight in mse
        after a number of steps, copy Q to Q_target.
        '''
        for e in range(set.epoches) :
            
            # click "NEW GAME"
            click(game_region[0] + game_region[2] * 0.66, game_region[1] + game_region[3] * 0.375)
            sleep(7)
            
            stateQueue = StateQueue()

            for n in range(set.steps_epoch) :
            
                # make action
                cur_shot = self.get_screenshot()
                if random.random() < max(set.eps_min, set.epsilon * (set.eps_decay ** (e + n / set.epoches))):
                    cur_action = random.randrange(set.actions_num)
                else :
                    cur_action = np.argmax(self.Q.predict(self.add_noise(cur_shot)))
                
                self.do_control(cur_action)
                
                nxt_shot = self.get_screenshot()
                cur_reward = stateQueue.calReward(cur_shot, nxt_shot)
                #print(cur_action, ",", cur_reward)
                
                stateQueue.addStep(cur_shot, cur_action, cur_reward, nxt_shot)
                
                if (stateQueue.getLength() > set.batch_size and n > set.train_thrshld and n % set.steps_train == 0) :
                    # Experience Replay
                    r = random.randint(set.batch_size, stateQueue.getLength())
                    input_scrshots, input_actions, rewards, train_nxt_shots = stateQueue.getStepsInArray(r - set.batch_size, r)
                    nxt_rewards = np.zeros((set.batch_size, set.actions_num))
                    
                    for j in range(set.batch_size) :
                        this_nxt_shots = np.reshape(train_nxt_shots[j], set.scrshot_shape)
                        #print(input_actions[j])
                        nxt_rewards[j, int(input_actions[j])] = np.argmax(self.Q_target.predict(self.add_noise(this_nxt_shots)))
                        
                    train_targets = np.add(np.reshape(rewards, (set.batch_size, 1)), np.multiply(set.gamma, nxt_rewards))
                    #print("train_targets:\n", train_targets.tolist())
                    
                    loss = self.Q.train_on_batch(input_scrshots, train_targets)
                    #print("loss: ", loss)
                    
                if n % set.steps_update_target == 0 and n != 0 :
                    print("assign Qtarget")
                    self.Q_target.set_weights(self.Q.get_weights())
                    self.Q_target.save_weights("Q_target_weight.h5")
                    
                # end for(STEP_PER_EPOCH)  
            print("end epoch", e)
            del stateQueue

            # Restart Game...
            # push ESC
            self.directInput.directKey("ESC")
            sleep(0.1)
            # click "QUIT"
            click(game_region[0] + game_region[2] * 0.13, game_region[1] + game_region[3] * 1.05)
            sleep(7)
            
        # end for(epoches)
        
        self.Q_target.save_weights("Q_target_weight.h5")
        
    # end def run
# end class Train

if __name__ == '__main__' :
    train = Train()
    train.count_down(3)
    train.run()
    

