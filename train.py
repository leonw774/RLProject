import sys
import random
import math
import numpy as np
from matplotlib import pyplot as plt
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
        self.model_optimizer = optimizers.rmsprop(lr = set.learning_rate, decay = 1e-6)
        
        self.Q = QNet(set.model_input_shape, set.actions_num)
        self.Q.summary()
        self.Q.compile(loss = "mse", optimizer = self.model_optimizer, metrics = ['mse'])
        
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
            
    def get_screenshot(self, wait_no_move = True, savefile = None) :
        # return screen-shot of game in np array in shape of set.shot_shape
        array_scrshot = np.zeros(set.shot_shape)
        cur = screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize)
        i = 0
        while(wait_no_move and i <= set.shot_wait_max) :
            #print("waiting for no moving")
            sleep(set.shot_intv_time)
            pre = cur
            cur = screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize)
            if np.sum(np.array(cur)) <= 256 : # is black
                sleep(1.6)
                continue
            if np.sum(np.absolute((np.array(pre) - np.array(cur)) / 256.0)) < 33 * set.no_move_thrshld :
                break
            i += 1
        
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
            slow_distance = 2400 # pixels
            fast_distance = 4000 # pixels
            slow_delta = 3 # pixels
            fast_delta = 25
        
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
            if id >= set.mouse_straight_angles :
                sleep(0.02)
        else :
            # is round
            id -= set.mouse_straight_angles * 2
            
            if id < set.mouse_round_angles * 2 :
                is_clockwise = 1
            else :
                is_clockwise = -1
                id -= set.mouse_round_angles * 2
            
            if id < set.mouse_round_angles : # slow
                radius = 540
                delta = 4
                proportion = 0.8
            else : # fast
                radius = 780
                delta = 20
                proportion = 0.67
            
            angles_divide = 36.0
            angle_bias = 4.0
            angle_offset = id / set.mouse_round_angles + angle_bias / angles_divide
            edge_leng = int(2 * (radius**2) * (1 - math.cos(1.0 / angles_divide))) 
            
            for i in range(int(angles_divide * proportion)) : 
                angle = 2 * math.pi * (i * is_clockwise / angles_divide + angle_offset)
                d_x = math.ceil(math.cos(angle) * delta)
                d_y = math.ceil(math.sin(angle) * delta)
                for j in range(edge_leng // delta) :
                    self.directInput.directMouse(d_x, d_y)
                    sleep(intv_time)
            sleep(0.01)
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
        while(1) : # somrtimes the game is not responsive to keybroad, you have to try more times
            shot1 = np.array(screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize))
            self.directInput.directKey("ESC")
            sleep(1)
            shot2 = np.array(screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize))
            if np.sum(np.abs(shot1 - shot2)) > set.no_move_thrshld : break
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
        loss_list = []
        end_reward_list = []
        averange_reward_list = []
        averange_Q_list = []
        logfile = file("log.csv", 'w')
        logfile.write("epoch, loss, end_reward, avg_reward, avg_Q")
        
        for e in range(set.epoches) :
            
            self.newgame()
            
            this_epoch_eps = max(set.eps_min, set.epsilon * (set.eps_decay ** e), random.random())
            avrg_reward = 0
            avrg_Q = 0
            avrg_Q_num = 0
            stuck_count = 0
            loss = 0

            for n in range(set.steps_epoch) :
                if (n + 1) % (set.steps_epoch / 10) == 0 :
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                cur_shot = self.get_screenshot(wait_no_move = False)

                # make action
                if random.random() < this_epoch_eps :
                    if n <= 1 :
                        cur_action = random.randrange(set.actions_num - 1)
                    else :
                        if set.use_p_normalizeation :
                            w = stepQueue.getActionsOccurrence()
                            w = w.max() - w
                            w /= w.sum()
                            cur_action = np.random.choice(np.arange(set.actions_num), p = w)
                        else :
                            cur_action = np.random.choice(np.arange(set.actions_num))
                else :
                    pred = self.Q.predict(self.add_noise(cur_shot))
                    cur_action = np.argmax(pred)
                    avrg_Q += np.max(pred)
                    avrg_Q_num += 1
                
                self.do_control(cur_action)
                
                nxt_shot = self.get_screenshot(wait_no_move = True)
                cur_reward = stepQueue.calReward(cur_shot, nxt_shot) # pre-action, after-action
                avrg_reward += cur_reward / set.steps_epoch
                #print(cur_action, ",", cur_reward)
                
                # check if stuck
                if set.check_stuck :
                    if cur_reward == 0 :
                        stuck_count += 1 
                    elif stuck_count > 0 :
                        stuck_count -= 1
                    if stuck_count > set.stuck_thrshld :
                        loss *= (float(set.steps_epoch) / float(n))
                        avrg_reward *= (float(set.steps_epoch) / float(n))
                        sys.stdout.write("at step " +  str(n) + " ")
                        sys.stdout.flush()
                        break
                
                if not set.ignore_zero_r or cur_reward != 0 or random.random() < min(set.ignore_zero_r_p_min, set.ignore_zero_r_p ** e) :
                    stepQueue.addStep(cur_shot, cur_action, cur_reward, nxt_shot)
                
                if (stepQueue.getLength() > set.train_thrshld) and n % set.steps_train == 0 :
                    # Experience Replay
                    random_step = random.randint(1, stepQueue.getLength() - set.train_size)
                    trn_cur_s, trn_a, trn_r, trn_nxt_s = stepQueue.getStepsAsArray(random_step, set.train_size)
                    
                    # make next predicted reward array and train input array at same time
                    new_r = np.zeros((set.train_size, set.actions_num))
                    for j in range(set.train_size) :
                        if set.steps_update_target > 0 :
                            new_r[j] = self.Q.predict(np.expand_dims(trn_cur_s[j], axis = 0))
                            predict_Q = self.Q.predict(np.expand_dims(trn_nxt_s[j], axis = 0))
                        else :
                            new_r[j] = self.Q_target.predict(np.expand_dims(trn_cur_s[j], axis = 0))
                            predict_Q = self.Q_target.predict(np.expand_dims(trn_nxt_s[j], axis = 0))
                        new_r[j, trn_a[j]] = trn_r[j] + np.max(predict_Q) * set.gamma
                        # Q_new = r + Q_predict(a,s) * gamma
                    
                    #print("new_r\n", new_r[0])
                    
                    loss += self.Q.train_on_batch(trn_cur_s, new_r)[0] / float(set.steps_epoch)
                    
                if set.steps_update_target > 0 and n + 1 % set.steps_update_target == 0 and n > set.train_thrshld :
                    #print("assign Qtarget")
                    self.Q_target.set_weights(self.Q.get_weights())
                    self.Q_target.save("Q_target_model.h5")
                    
            # end for(STEP_PER_EPOCH)
            end_reward = stepQueue.getCurMap(cur_shot)
            if avrg_Q_num > 0 : avrg_Q /= float(avrg_Q_num)
            
            print("\tend %d\tat map %d\tloss: %.4f" % (e, end_reward, loss))
            logfile.write(str(loss) + "," + str(avrg_reward) + "," + str(avrg_reward) + "," + str(avrg_Q))
            loss_list.append(loss)
            end_reward_list.append(end_reward)
            averange_reward_list.append(avrg_reward)
            averange_Q_list.append(avrg_Q)
            
            #stepQueue.clear()
            self.Q_target.save("Q_target_model.h5")
            
            # Restart Game...
            self.quitgame()
            
        # end for(epoches)
        
        plt.figure(figsize = (10, 6))
        plt.xlabel("epoch")
        plt.ylim(0, 2)
        plt.plot(loss_list, label = "loss")
        plt.savefig("loss_fig.png")
        plt.close()
        
        plt.figure(figsize = (10, 6))
        plt.xlabel("epoch")
        plt.plot(end_reward_list, label = "end reward")
        plt.plot(end_reward_list, label = "averange reward")
        plt.savefig("reward_fig.png")
        plt.close()
        
        plt.figure(figsize = (10, 6))
        plt.xlabel("epoch")
        plt.plot(averange_Q_list, label = "averange Q")
        plt.savefig("Q_fig.png")
        plt.close()
        
        self.Q_target.save("Q_target_model.h5")
        
    # end def fit
    
    def test(self, model_weight_name, rounds = 1) :
        print("test begin for:", model_weight_name)
        self.Q = load_model(model_weight_name)
        end_map_list = []
        
        for i in range(rounds) :
        
            # click "NEW GAME"
            self.newgame()
            stepQueue = StepQueue()
            cur_shot = self.get_screenshot() 
            
            for n in range(set.steps_test) :
                cur_shot = self.get_screenshot()
                
                predict_Q = np.squeeze(self.Q.predict(self.add_noise(cur_shot)))
                if predict_Q.sum() <= 0.01 :
                    cur_action = random.randrange(set.actions_num)
                elif random.random() <= set.eps_test :
                    w = predict_Q
                    w[w < 0] = 0.0
                    w = np.exp(w) - 1
                    w /= w.sum()
                    cur_action = np.random.choice(np.arange(set.actions_num), p = w)
                else :
                    cur_action = np.argmax(predict_Q)
                #print(predict_Q)
                print("at map", stepQueue.getCurMap(cur_shot), "choose", cur_action, "with Q:", predict_Q[cur_action])
                    
                self.do_control(cur_action)
            # end for step_test
            
            end_map = stepQueue.getCurMap(cur_shot)
            end_map_list.append(end_map)
            
            print("test end\tat map: ", end_map)
            #screenshot(region = self.GameRegion).save("test_scrshot.png")
        
            # Exit Game...
            self.quitgame()
        
        print(end_map_list)

    # end def test
    
    def random_action(self, steps = None) :
        # click "NEW GAME"
        self.newgame()
        stepQueue = StepQueue()
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
        
        del stepQueue
        print("test end, of reward: %.2f", cur_reward)
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
    train.test("Q_target_model.h5", 20)
    

