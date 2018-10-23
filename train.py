import sys
import random
import math
import numpy as np
from PIL import Image
from time import sleep
from datetime import datetime, timedelta
from pyautogui import click, screenshot
from keras.models import Model, load_model, optimizers

from setting import TrainingSetting as set
from stepqueue import StepQueue
from directinputs import Keys
from QNet import QNet

class Train() :
    
    def __init__(self, use_weight_file = None) :
        
        self.GameRegion = set.get_game_region("Getting Over It")
        self.directInput = Keys()
        self.model_optimizer = optimizers.sgd(lr = 0.1)
        
        self.Q = QNet(set.model_input_shape, set.actions_num)
        self.Q.summary()
        self.Q.compile(loss = "mse", optimizer = self.model_optimizer)
        
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
    
    def count_down(self, cd) :
        for i in range(cd) :
            print(cd - i)
            sleep(1.0)
            
    def get_screenshot(self, num = 1, savefile = None) :
        array_scrshot = np.zeros(set.shot_shape)
        cur = screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize, resample = Image.NEAREST)
        i = 0
        while(i <= set.shot_wait_max) :
            i += 1
            #print("waiting for no moving")
            sleep(set.shot_intv_time)
            pre = cur
            cur = screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize, resample = Image.NEAREST)
            if np.sum(np.array(cur)) < 256 :
                continue
            #print(np.sum(np.absolute((np.array(pre) - np.array(cur)) / 256.0)))
            if np.sum(np.absolute((np.array(pre) - np.array(cur)) / 256.0)) < 32 * set.no_move_thrshld :
                break
        
        if set.shot_c == 1 :
            array_scrshot = np.reshape(np.array(cur.convert('L')) / 255.5, set.shot_shape)
        elif set.shot_c == 3 :
            array_scrshot[0] = np.array(cur) / 255.5
        else :
            raise Exception("shot_c isn't right.")
        if savefile : scrshot.save(savefile)
        return array_scrshot
    # end def get_screenshot
    
    def add_noise(self, noisy_scrshot) :
        noisy_scrshot += np.random.uniform(low = -set.noise_range, high = set.noise_range, size = noisy_scrshot.shape)
        noisy_scrshot[noisy_scrshot > 1.0] = 1.0
        noisy_scrshot[noisy_scrshot < 0.0] = 0.0
        return noisy_scrshot
    # end def get_screen_rect
    
    def do_control(self, id) :
        
        intv_time = 0.001
        
        if id < set.mouse_straight_angles * 2 :
            # is straight
            slow_distance = 3000 # pixels
            fast_distance = 5000 # pixels
            slow_delta = 4 # pixels
            fast_delta = 32
        
            if id < set.mouse_straight_angles :
                delta, distance = slow_delta, slow_distance
            else :
                delta, distance = fast_delta, fast_distance
            
            angle = 2 * math.pi * id / set.mouse_straight_angles
            d_x = math.ceil(math.cos(angle) * delta)
            d_y = math.ceil(math.sin(angle) * delta)
            
            for i in range(distance // delta) :
                self.directInput.directMouse(d_x, d_y)
                sleep(intv_time)
        else :
            # is round
            id -= set.mouse_straight_angles * 2
            radius = 600
            d_angle_ratio = 36
            v = int(2 * (radius**2) * (1 - math.cos(1.0 / d_angle_ratio))) 
            delta = 8
            if id < set.mouse_round_angles :
                delta = -delta
                d_angle_ratio = -d_angle_ratio
            
            for i in range(int(d_angle_ratio * 0.8)) : 
                angle = 2 * math.pi * (id / set.mouse_round_angles + i / float(d_angle_ratio))
                d_x = math.ceil(math.cos(angle) * delta)
                d_y = math.ceil(math.sin(angle) * delta)
                for j in range(v // delta + 1) :
                    self.directInput.directMouse(d_x, d_y)
                    sleep(intv_time)
        
        sleep(set.do_control_pause)
    # end def do_control()
    
    def newgame(self) :
        sleep(1)
        # click "NEW GAME"
        click(self.GameRegion[0] + self.GameRegion[2] * 0.70, self.GameRegion[1] + self.GameRegion[3] * 0.40)
        sleep(8)
    
    def quitgame(self) :
        sleep(1)
        # push ESC
        self.directInput.directKey("ESC")
        sleep(1)
        # click "QUIT"
        click(self.GameRegion[0] + self.GameRegion[2] * 0.15, self.GameRegion[1] + self.GameRegion[3] * 1.05)
        sleep(10)
    
    def fit(self) :
        '''
        We will train Q, at time t, as:
            y_pred = Q([state_t, a_t])
            y_true = r_t + gamma * max(Q_target([state_t, for a in A]))
        update Q's weight in mse
        after a number of steps, copy Q to Q_target.
        '''
        
        stepQueue = StepQueue()
        
        for e in range(set.epoches) :
            
            self.newgame()
            
            total_reward = 0
            no_reward_count = 0
            this_epoch_epsilon = max(set.eps_min, set.epsilon * (set.eps_decay ** e), random.random())
            input_shots = np.zeros((1, set.shot_h, set.shot_w, set.shot_c * set.shot_n))

            for n in range(set.steps_epoch) :
                if (n + 1) % (set.steps_epoch / 10) == 0 :
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                cur_shot = self.get_screenshot()
                input_shots[:,:,:, : -set.shot_c] = input_shots[:,:,:, set.shot_c : ] # dequeue
                input_shots[:,:,:, -set.shot_c : ] = cur_shot # enqueue

                # make action
                if n <= set.shot_n or random.random() < this_epoch_epsilon :
                    if n == 0 :
                        cur_action = random.randint(0, set.actions_num - 1)
                    else :
                        p_weight = n - stepQueue.getActionsOccurrence()
                        p_weight = p_weight / p_weight.sum()
                        cur_action = np.random.choice(np.arange(set.actions_num), p = p_weight)
                else :
                    cur_action = np.argmax(self.Q.predict(self.add_noise(input_shots)))
                
                self.do_control(cur_action)
                
                nxt_shot = self.get_screenshot()
                cur_reward = stepQueue.calReward(cur_shot, nxt_shot) # pre-action, after-action
                #print(cur_action, ",", cur_reward)
                if cur_reward == "stuck" :
                    sys.stdout.write(" at step " +  str(n) + ", ")
                    sys.stdout.flush()
                    screenshot(region = self.GameRegion).save("stuck_at_epoch" + str(e) + ".png")
                    break
                
                total_reward += cur_reward
                
                if set.no_reward_break :
                    if cur_reward <= set.bad_r :
                        no_reward_count += 1
                    else :
                        no_reard_count = 0
                    if n > set.train_thrshld and no_reward_count * 2 > n :
                        break
                
                stepQueue.addStep(cur_shot, cur_action, cur_reward, nxt_shot)
                
                if stepQueue.getLength() > set.train_size and n > set.train_thrshld and n % set.steps_train == 0 :
                    # Experience Replay
                    random_step = random.randint(set.shot_n + 1, stepQueue.getLength() - set.train_size - 1)
                    train_cur_shots, train_actions, train_rewards, train_nxt_shots = stepQueue.getStepsAsArray(random_step, set.train_size)
                    train_input_shots = np.zeros((set.train_size, set.shot_h, set.shot_w, set.shot_c * set.shot_n))
                    
                    # make replay reward array
                    replay_rewards = np.zeros((set.train_size, set.actions_num))
                    for j in range(set.train_size) :
                        replay_rewards[j, int(train_actions[j])] = train_rewards[j]
                    
                    # make next predicted reward array and train input array at same time
                    nxt_rewards = np.zeros((set.train_size, set.actions_num))
                    for j in range(set.train_size) :
                        if j < set.shot_n - 1 :
                            train_input_shots[j,:,:, : -set.shot_c] = stepQueue.getShotsAsArray(random_step - set.shot_n, set.shot_n - 1)
                        else :
                            train_input_shots[j,:,:, : -set.shot_c] = train_input_shots[j - 1,:,:, set.shot_c : ]
                        train_input_shots[j,:,:, -set.shot_c : ] = train_cur_shots[j]
                        train_input_shots[j] = self.add_noise(train_input_shots[j])
                        this_train_input_shots = np.expand_dims(train_input_shots[j], axis = 0)
                        if set.steps_update_target > 0 :
                            nxt_rewards[j] = self.Q.predict(this_train_input_shots)
                        else :
                            nxt_rewards[j] = self.Q_target.predict(this_train_input_shots)
                        nxt_rewards[j, int(train_actions[j])] *= set.gamma
                        
                    train_targets_rewards = replay_rewards + nxt_rewards
                    
                    #print("replay_rewards\n", replay_rewards[0])
                    #print("nxt_rewards\n", nxt_rewards[0])
                    #print("train_targets\n", train_targets[0])
                    
                    loss = self.Q.train_on_batch(train_input_shots, train_targets_rewards)
                    #print("loss: ", loss)
                    
                if set.steps_update_target > 0 and n % set.steps_update_target == 0 and n > set.train_thrshld :
                    #print("assign Qtarget")
                    self.Q_target.set_weights(self.Q.get_weights())
                    self.Q_target.save("Q_target_model.h5")
                    
            # end for(STEP_PER_EPOCH)
            
            print("end epoch", e, "total_reward:", total_reward)
            stepQueue.clear()
            self.Q_target.save("Q_target_model.h5")
            # Restart Game...
            self.quitgame()
            
        # end for(epoches)
        
        self.Q_target.save("Q_target_model.h5")
        
    # end def fit
    
    def eval(self, model_weight_name) :
        print("eval begin for:", model_weight_name)
        self.Q = load_model(model_weight_name)
        
        # click "NEW GAME"
        self.newgame()
        
        stepQueue = StepQueue()
        total_reward = 0
        input_shots = np.zeros((1, set.shot_h, set.shot_w, set.shot_c * set.shot_n))
        for n in range(set.steps_test) :
            cur_shot = self.get_screenshot()
            input_shots[:,:,:, : -set.shot_c] = input_shots[:,:,:, set.shot_c: ] # dequeue
            input_shots[:,:,:, -set.shot_c : ] = cur_shot # enqueue
            
            if n <= set.shot_n or random.random() < set.eps_test :
                cur_action = random.randrange(set.actions_num)
            else :
                predict_Q = self.Q.predict(self.add_noise(input_shots))
                #print(predict_Q)
                cur_action = np.argmax(predict_Q)
                print("choose", cur_action, "with max Q:", np.max(predict_Q))
                
            self.do_control(cur_action)
            nxt_shot = self.get_screenshot()
            cur_reward = stepQueue.calReward(cur_shot, nxt_shot) # pre, cur
            if cur_reward == "stuck" : break
            stepQueue.addStep(cur_shot, cur_action, cur_reward, nxt_shot)
            total_reward += cur_reward
        
        del stepQueue
        print("eval end, total_reward:", total_reward)
        screenshot(region = self.GameRegion).save("eval_scrshot.png")
        
        # Exit Game...
        self.quitgame()

    # end def eval
    
    def random_action(self, steps = None) :
        # click "NEW GAME"
        self.newgame()
        stepQueue = StepQueue()
        total_reward = 0
        if steps == None : steps = set.steps_test
        for n in range(steps) :
            cur_shot = self.get_screenshot()
            cur_action = random.randrange(set.actions_num)
            self.do_control(cur_action)
            nxt_shot = self.get_screenshot()
            cur_reward = stepQueue.calReward(cur_shot, nxt_shot)
            print(cur_action, ",", cur_reward)
            if cur_reward == "stuck" :
                print("at step", n)
                screenshot(region = self.GameRegion).save("stuck_at_random.png")
                break
            stepQueue.addStep(cur_shot, cur_action, cur_reward, nxt_shot)
            total_reward += cur_reward
        
        del stepQueue
        print("eval end, totalReward:", total_reward)
        # Exit Game...
        self.quitgame()
    # end def
    
# end class Train

if __name__ == '__main__' :
    train = Train()
    train.count_down(3)
    starttime = datetime.now()
    #train.random_action()
    train.fit()
    print(datetime.now() - starttime)
    train.eval("Q_target_model.h5")
    

