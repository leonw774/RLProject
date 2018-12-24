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

class DQN() :
    
    def __init__(self, use_weight_file = None) :
        
        self.GameRegion = set.get_game_region("Getting Over It")
        self.directInput = Keys()
        self.model_optimizer = optimizers.rmsprop(lr = set.learning_rate, decay = set.learning_rate_decay)
        
        self.Q = QNet(set.model_input_shape, set.actions_num)
        self.Q.summary()
        self.Q.compile(loss = "mse", optimizer = self.model_optimizer, metrics = ['mse'])
        
        if use_weight_file :
            self.Q.load_weights(use_weight_file)
        
        if set.use_target_Q :
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
        if savefile : cur.save(savefile)
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
            fast_delta = 20
        
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
                radius = 560
                delta = 4
                proportion = 0.8
            else : # fast
                radius = 720
                delta = 20
                proportion = 0.7
            
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
        while(1) : # somrtimes the game is not responsive to keybroad, you have to try more times
            shot1 = np.array(screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize))
            click(self.GameRegion[0] + self.GameRegion[2] * 0.70, self.GameRegion[1] + self.GameRegion[3] * 0.40)
            sleep(0.2)
            shot2 = np.array(screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize))
            if np.sum(np.abs(shot1 - shot2)) > set.no_move_thrshld : break
        sleep(8)
    
    def quitgame(self) :
        sleep(1)
        # push ESC
        while(1) : # somrtimes the game is not responsive to keybroad, you have to try more times
            shot1 = np.array(screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize))
            self.directInput.directKey("ESC")
            sleep(0.2)
            shot2 = np.array(screenshot(region = self.GameRegion).convert('RGB').resize(set.shot_resize))
            if np.sum(np.abs(shot1 - shot2)) > set.no_move_thrshld : break
        # click "QUIT"
        click(self.GameRegion[0] + self.GameRegion[2] * 0.15, self.GameRegion[1] + self.GameRegion[3] * 1.05)
        sleep(10)
        
    def draw_fig(self, loss, avgQ, avgR, endMap, testEndMap) :
    
        plt.figure(figsize = (12, 8))
        plt.xlabel("epoch")
        plt.ylim(0, 0.2)
        plt.plot(loss, label = "loss")
        plt.legend(loc = "upper right")
        plt.savefig("fig/loss.png")
        plt.close()
        
        plt.figure(figsize = (12, 6))
        plt.xlabel("epoch")
        plt.plot(avgR, label = "averange reward")
        plt.legend(loc = "upper right")
        plt.savefig("fig/avg_reward.png")
        plt.close()
        
        plt.figure(figsize = (12, 6))
        plt.xlabel("epoch")
        plt.plot(avgQ, label = "averange Q")
        plt.legend(loc = "upper right")
        plt.savefig("fig/avg_Q.png")
        plt.close()
        
        the_tree = [11] * len(endMap)
        
        plt.figure(figsize = (12, 6))
        plt.xlabel("epoch")
        plt.ylim(0, 60)
        plt.plot(endMap, label = "train_end_map")
        plt.plot(the_tree, label = "The Tree")
        plt.legend(loc = "upper right")
        plt.savefig("fig/train_end_map.png")
        plt.close()
        
        the_tree = [11] * len(testEndMap)
        
        plt.figure(figsize = (12, 6))
        plt.xlabel("test#")
        plt.ylim(0, 60)
        plt.plot(testEndMap, label = "test_end_map")
        plt.plot(the_tree, label = "The Tree")
        plt.legend(loc = "upper right")
        plt.savefig("fig/test_end_map.png")
        plt.close()
        
    def test(self, model_weight_name, rounds = 1, max_step = set.steps_test, goal = None, verdict = False) :
        if verdict : print("test begin for:", model_weight_name)
        testQ = load_model(model_weight_name)
        if goal is not None :
            test_result = np.zeros((rounds, 2))
        else :
            test_result = np.zeros((rounds))
        
        for i in range(rounds) :
        
            # click "NEW GAME"
            self.newgame()
            test_stepQueue = StepQueue()
            cur_shot = self.get_screenshot(wait_no_move = True) 
            
            for n in range(max_step) :
                cur_shot = self.get_screenshot()
                
                if goal is not None :
                    if test_stepQueue.getCurMap(cur_shot) >= goal :
                        print("reached goal!")
                        test_result[i] = np.array((n, True))
                        break
                
                predict_Q = np.squeeze(testQ.predict(self.add_noise(cur_shot)))
                if predict_Q.sum() < -0.01 and predict_Q.sum() > -0.01 :
                    cur_action = random.randrange(set.actions_num)
                elif random.random() <= set.eps_test :
                    # soft max
                    w = predict_Q
                    w = np.exp(w)
                    w /= w.sum()
                    cur_action = np.random.choice(np.arange(set.actions_num), p = w)
                else :
                    cur_action = np.argmax(predict_Q)
                #print(predict_Q)
                if verdict : print("at map", test_stepQueue.getCurMap(cur_shot), "choose", cur_action, "with Q:", predict_Q[cur_action])
                    
                self.do_control(cur_action)
                
            # end for step_test
            
            end_map = test_stepQueue.getCurMap(cur_shot)
            
            if goal is not None :
                if end_map >= goal and test_result[i, 0] == 0:
                    print("reached goal!")
                    test_result[i] = np.array((max_step, True))
                else :
                    test_result[i] = np.array((max_step, False))
            else :
                test_result[i] = end_map
            
            if verdict : print("test round ", i, " end\tat map: ", end_map)
        
            # Exit Game...
            self.quitgame()
        return test_result
    # end def test
    
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
        endMap_list = []
        avgR_list = []
        avgQ_list = []
        test_endMap_list = []
        
        logfile = open("log.csv", 'w')
        logfile.write("epoch, loss, endMap, avrgR, avrgQ\n")
        
        for e in range(set.epoches) :
            
            self.newgame()
            sys.stdout.write(str(e) + ":")
            sys.stdout.flush()
            
            nxt_shot = self.get_screenshot(wait_no_move = False)
            
            this_epoch_eps = max(set.eps_min, set.epsilon * (set.eps_decay ** e), random.random())
            avrgR = 0
            avrgQ = 0
            Q_count = 0
            stuck_count = 0
            predQ = 0
            loss = 0

            for n in range(set.steps_epoch) :
                if (n + 1) % (set.steps_epoch / 10) == 0 :
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                cur_shot = nxt_shot

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
                    predQ = np.squeeze(self.Q.predict(self.add_noise(cur_shot)))
                    cur_action = np.argmax(predQ)
                    avrgQ += np.max(predQ)
                    Q_count += 1
                
                self.do_control(cur_action)
                
                nxt_shot = self.get_screenshot(wait_no_move = True)
                cur_reward = stepQueue.calReward(cur_shot, nxt_shot) # pre-action, after-action
                avrgR += cur_reward / set.steps_epoch
                #print(cur_action, ",", cur_reward)
                
                # check if stuck
                if set.check_stuck :
                    if cur_reward == 0 :
                        stuck_count += 1 
                    elif stuck_count > 0 :
                        stuck_count -= 1
                    if stuck_count > set.stuck_thrshld :
                        loss *= (float(set.steps_epoch) / float(n))
                        avrgR *= (float(set.steps_epoch) / float(n))
                        sys.stdout.write(str(n))
                        sys.stdout.flush()
                        break
                        
                if not set.ignore_zero_r or cur_reward != 0 or random.random() < min(set.ignore_zero_r_p_min, set.ignore_zero_r_p ** e) :
                    stepQueue.addStep(cur_shot, cur_action, cur_reward)
                
                if (stepQueue.getLength() > set.train_thrshld) and n % set.steps_train == 0 :
                    # Experience Replay
                    random_step = random.randint(0, stepQueue.getLength() - set.train_size - 1)
                    trn_cur_s, trn_a, trn_r = stepQueue.getStepsAsArray(random_step, set.train_size + 1)
                    
                    # make next predicted reward array and train input array at same time
                    new_r = np.zeros((set.train_size, set.actions_num))
                    for j in range(set.train_size) :
                        if not set.use_target_Q :
                            new_r[j] = self.Q.predict(np.expand_dims(trn_cur_s[j], axis = 0))
                            predict_Q = self.Q.predict(np.expand_dims(trn_cur_s[j+1], axis = 0))
                        else :
                            new_r[j] = self.Q_target.predict(np.expand_dims(trn_cur_s[j], axis = 0))
                            predict_Q = self.Q_target.predict(np.expand_dims(trn_cur_s[j+1], axis = 0))
                        new_r[j, trn_a[j]] = trn_r[j] + np.max(predict_Q) * set.gamma
                        # Q_new = r + Q_predict(a,s) * gamma
                    
                    #print("new_r\n", new_r[0])
                    
                    loss += self.Q.train_on_batch(trn_cur_s[:set.train_size], new_r)[0] / float(set.steps_epoch)
                    
            # end for(STEP_PER_EPOCH)
            self.quitgame()
            
            endMap = stepQueue.getCurMap(cur_shot)
            if Q_count > 0 :
                avrgQ /= float(Q_count)
            
            # write log file and log list
            print("\tend at map %d\tloss: %.4f  avrgQ: %.3f avrgR: %.3f" % (endMap, loss, avrgQ, avrgR))
            log_string = str(e) + "," + str(loss) + "," + str(endMap) + "," + str(avrgR) + "," + str(avrgQ) + "\n"
            logfile.write(log_string)
            loss_list.append(loss)
            endMap_list.append(endMap)
            avgR_list.append(avrgR)
            avgQ_list.append(avrgQ)
            
            if set.use_target_Q & stepQueue.getLength() > set.train_thrshld :
                self.Q_target.set_weights(self.Q.get_weights())
                self.Q_target.save("Q_model.h5")
            else :
                self.Q.save("Q_model.h5")
                
            if (e + 1) % set.test_intv == 0 :
                test_endMap = self.test("Q_model.h5", verdict = False)[0]
                test_endMap_list.append(test_endMap);
                print("test: ", test_endMap)
            
            if (e + 1) % set.draw_fig_intv == 0 :
                self.draw_fig(loss_list, avgR_list, avgQ_list, endMap_list, test_endMap_list)
            
        # end for(epoches)
        self.draw_fig(loss_list, avgR_list, avgQ_list, endMap_list, test_endMap_list)
        
    # end def fit
    
    def random_action(self, steps = None) :
        # click "NEW GAME"
        self.newgame()
        random_stepQueue = StepQueue()
        if steps == None : steps = set.steps_test
        for n in range(steps) :
            cur_shot = self.get_screenshot(savefile = ("output/" + str(n) + ".png"))
            cur_action = random.randrange(set.actions_num)
            self.do_control(cur_action)
            nxt_shot = self.get_screenshot()
            cur_reward = random_stepQueue.calReward(cur_shot, nxt_shot)
            print(cur_action, ",", cur_reward)
            if cur_reward == "stuck" :
                print("at step", n)
                screenshot(region = self.GameRegion).save("stuck_at_random.png")
                break
        
        del random_stepQueue
        print("test end, of reward: %.2f", cur_reward)
        # Exit Game...
        self.quitgame()
    # end def
    
# end class DQN
    
    