import sys
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from keras.models import Model, load_model, optimizers

from setting import Setting as set
from stepqueue import StepQueue
from qnet import QNet
from gameagent import GameAgent

class DQN() :
    
    def __init__(self, use_weight_file = None, use_target_Q = False) :
        
        self.game = GameAgent()
        self.model_optimizer = optimizers.rmsprop(lr = set.learning_rate, decay = set.learning_rate_decay)
        
        self.Q = QNet(set.model_input_shape, set.actions_num)
        self.Q.summary()
        self.Q.compile(loss = "mse", optimizer = self.model_optimizer, metrics = ['mse'])
        
        if use_weight_file :
            self.Q.load_weights(use_weight_file)
        
        if use_target_Q :
            self.Q_target = QNet(set.model_input_shape, set.actions_num)
            self.Q_target.set_weights(self.Q.get_weights())
        else :
            self.Q_target = None
            
        self.A = []
        for i in range(set.actions_num) :
            action_onehot = np.zeros((1, set.actions_num))
            action_onehot[0, i] = 1
            self.A.append(action_onehot)
    
    def count_down(self, cd) :
        for i in range(cd) :
            print(cd - i)
            sleep(1.0)
    
    def add_noise(self, noisy_scrshot) :
        noisy_scrshot += np.random.uniform(low = -set.noise_range, high = set.noise_range, size = noisy_scrshot.shape)
        noisy_scrshot[noisy_scrshot > 1.0] = 1.0
        noisy_scrshot[noisy_scrshot < 0.0] = 0.0
        return noisy_scrshot
    # end def get_screen_rect
    
    def draw_fig(self, loss, avgQ, avgR, endMap, testEndMap) :
    
        plt.figure(figsize = (12, 8))
        plt.xlabel("epoch")
        plt.ylim(0, 0.5)
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
            self.game.newgame()
            test_stepQueue = StepQueue()
            cur_shot = self.game.get_screenshot(wait_no_move = True) 
            
            for n in range(max_step) :
                cur_shot = self.game.get_screenshot()
                
                if goal is not None :
                    if test_stepQueue.getCurMap(cur_shot) >= goal :
                        if verdict : print("Reached Goal!")
                        test_result[i] = np.array((n, True))
                        break
                
                predict_Q = np.squeeze(testQ.predict(self.add_noise(cur_shot)))
                if np.random.random() < set.eps_test :
                    # soft max
                    w = predict_Q
                    w = np.exp(w)
                    w /= w.sum()
                    cur_action = np.random.choice(np.arange(set.actions_num), p = w)
                else :
                    cur_action = np.argmax(predict_Q)
                #print(predict_Q)
                if verdict : print("Score:", test_stepQueue.getCurMap(cur_shot), "\nDo action", cur_action, "with Q:", predict_Q[cur_action], "\n")
                    
                self.game.do_control(cur_action)
                
            # end for step_test
            
            end_map = test_stepQueue.getCurMap(cur_shot)
            
            if goal is not None :
                if end_map >= goal and test_result[i, 0] == 0:
                    if verdict : print("Reached Goal!")
                    test_result[i] = np.array((max_step, True))
                else :
                    test_result[i] = np.array((max_step, False))
            else :
                test_result[i] = end_map
            
            if verdict : print("Test round", i, "ended. Score:", end_map, "\n")
        
            # Exit Game...
            self.game.quitgame()
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
            
            self.game.newgame()
            sys.stdout.write(str(e) + ":")
            sys.stdout.flush()
            
            nxt_shot = self.game.get_screenshot(wait_no_move = False)
            
            this_epoch_eps = max(set.eps_min, set.epsilon * (set.eps_decay ** e))
            avgR_list.append(0)
            avgQ_list.append(0)
            loss_list.append(0)
            stuck_count = 0

            for n in range(set.steps_epoch) :
                if (n + 1) % (set.steps_epoch / 10) == 0 :
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                cur_shot = nxt_shot

                # make action
                predQ = np.squeeze(self.Q.predict(self.add_noise(cur_shot)))
                avgQ_list[e] += np.max(predQ) / set.steps_epoch
                if np.random.random() < this_epoch_eps :
                    cur_action = np.random.randint(set.actions_num)
                else :
                    cur_action = np.argmax(predQ)
                
                self.game.do_control(cur_action)
                
                nxt_shot = self.game.get_screenshot(wait_no_move = True)
                cur_reward = stepQueue.calReward(cur_shot, nxt_shot) # pre-action, after-action
                avgR_list[e] += cur_reward / set.steps_epoch
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
                
                if stepQueue.getLength() > set.train_thrshld and n % set.steps_train == 0 :
                    # Experience Replay
                    random_step = random.randint(stepQueue.getLength() - (set.train_size + 1))
                    trn_s, trn_a, trn_r = stepQueue.getStepsAsArray(random_step, set.train_size + 1)
                    
                    # make next predicted reward array and train input array at same time
                    new_r = np.zeros((set.train_size, set.actions_num))
                    for j in range(set.train_size) :
                        if self.Q_target == None :
                            new_r[j] = self.Q.predict(np.expand_dims(trn_s[j], axis = 0))
                            predict_Q = self.Q.predict(np.expand_dims(trn_s[j+1], axis = 0))
                        else :
                            new_r[j] = self.Q_target.predict(np.expand_dims(trn_s[j], axis = 0))
                            predict_Q = self.Q_target.predict(np.expand_dims(trn_s[j+1], axis = 0))
                        new_r[j, trn_a[j]] = trn_r[j] + np.max(predict_Q) * set.gamma
                        # Q_new = r + Q_predict(a,s) * gamma
                    
                    #print("new_r\n", new_r[0])
                    
                    loss_list[e] += self.Q.train_on_batch(trn_s[:set.train_size], new_r)[0] / set.steps_epoch * set.steps_train
                    
            # end for(STEP_PER_EPOCH)
            self.game.quitgame()
            
            endMap = stepQueue.getCurMap(cur_shot)
            
            # write log file and log list
            print("\tend at map %d\tloss: %.4f  avrgQ: %.3f avrgR: %.3f" % (endMap, loss, avrgQ, avrgR))
            log_string = ','.join([str(e), str(loss_list[e]), str(endMap), str(avrgR_list[e]), str(avrgQ_list[e])])
            logfile.write(log_string + "\n")
            loss_list.append(loss)
            endMap_list.append(endMap)
            avgQ_list.append(avrgQ)
            
            if self.Q_target != None and stepQueue.getLength() > set.train_thrshld :
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
        self.game.newgame()
        random_stepQueue = StepQueue()
        if steps == None : steps = set.steps_test
        for n in range(steps) :
            cur_shot = self.game.get_screenshot(savefile = ("output/" + str(n) + ".png"))
            cur_action = random.randrange(set.actions_num)
            self.game.do_control(cur_action)
            nxt_shot = self.game.get_screenshot()
            cur_reward = random_stepQueue.calReward(cur_shot, nxt_shot)
            print(cur_action, ",", cur_reward)
            if cur_reward == "stuck" :
                print("at step", n)
                screenshot(region = self.GameRegion).save("stuck_at_random.png")
                break
        
        del random_stepQueue
        print("test end, of reward: %.2f", cur_reward)
        # Exit Game...
        self.game.quitgame()
    # end def
    
# end class DQN
    
    