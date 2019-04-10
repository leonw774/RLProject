import sys
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from keras.models import Model, load_model, optimizers

from configure import Configuration as cfg
from stepqueue import StepQueue
from qnet import QNet
from gameagent import GameAgent
from gamereward import DiffReward, TieredDiffReward, MapReward

class DQN() :
    
    def __init__(self) :
        self.logfile = open("log.csv", 'w')
        self.game = GameAgent()
        self.model_optimizer = optimizers.rmsprop(lr = cfg.learning_rate, decay = cfg.learning_rate_decay)
            
        self.A = []
        for i in range(cfg.actions_num) :
            action_onehot = np.zeros((1, cfg.actions_num))
            action_onehot[0, i] = 1
            self.A.append(action_onehot)
    
    def countdown(self, cd) :
        for i in range(cd) :
            print(cd - i)
            sleep(1.0)
    
    def addNoise(self, noisy_scrshot) :
        noisy_scrshot += np.random.uniform(low = -cfg.noise_range, high = cfg.noise_range, size = noisy_scrshot.shape)
        noisy_scrshot[noisy_scrshot > 1.0] = 1.0
        noisy_scrshot[noisy_scrshot < 0.0] = 0.0
        return noisy_scrshot
    # end def get_screen_rect
    
    def writeLog(self, epoch, loss, avgQ, avrR, endMap) :
        log_string = ','.join([str(epoch), str(loss), str(avgQ), str(avrR), str(endMap)])
        self.logfile.write(log_string + "\n")
    
    def drawFig(self, loss, avgQ, avgR, endMap = None, testEndMap = None) :
    
        plt.figure(figsize = (12, 8))
        plt.xlabel("epoch")
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
        
    def test(self, model_weight_name, epsilon = cfg.eps_test, rounds = 1, max_step = cfg.steps_test, goal = None, verdict = False) :

        testQ = load_model(model_weight_name)
        if verdict :
            print("Test begin for:", model_weight_name, "Step Limit:", max_step)    
        
        if goal :
            if verdict : print("Goal:", goal)
            test_result = np.zeros((rounds, 2))
        else :
            test_result = np.zeros((rounds))
        
        for i in range(rounds) :
            if verdict : print("\nRound", i, "begin")
            # click "NEW GAME"
            self.game.newgame()
            test_rewardFunc = MapReward()
            cur_shot = self.game.getScreenshot(wait_no_move = True) 
            
            for n in range(max_step) :
                predict_Q = np.squeeze(testQ.predict(self.addNoise(cur_shot)))
                if np.random.random() < epsilon :
                    # soft max
                    w = predict_Q
                    w = np.exp(w)
                    w /= w.sum()
                    cur_action = np.random.choice(np.arange(cfg.actions_num), p = w)
                else :
                    cur_action = np.argmax(predict_Q)
                #print(predict_Q)
                if verdict : print("Score:", test_rewardFunc.getCurMap(cur_shot), "\nDo action", cur_action, "with Q:", predict_Q[cur_action], "\n")
                    
                self.game.doControl(cur_action)
                
                cur_shot = self.game.getScreenshot()
                
                if goal :
                    if test_rewardFunc.getCurMap(cur_shot) >= goal :
                        if verdict : print("Reached Goal!")
                        test_result[i] = np.array((n, True))
                        break
                
            # end for step_test
            
            # check if reached goal
            if goal :
                if test_result[i, 0] == 0 : # if not reached goal
                    print("Time out")
                    test_result[i] = np.array((max_step, False))
            else :
                test_result[i] = test_rewardFunc.getCurMap(self.game.getScreenshot())
            
            if verdict : print("Test round", i, "ended. Score:", end_map, "\n")
        
            # Exit Game...
            self.game.quitgame()
        return test_result
    # end def test
    
    def fit(self, load_weight_name = None, save_weight_name = "Q_model.h5", use_target_Q = False) :
        '''
        We will train Q, at time t, as:
            y_pred = Q([state_t, a_t])
            y_true = r_t + gamma * max(Q_target([state_t, for a in A]))
        update Q's weight in mse
        after a number of steps, copy Q to Q_target.
        '''
        
        Q = QNet(cfg.model_input_shape, cfg.actions_num)
        Q.summary()
        Q.compile(loss = "mse", optimizer = self.model_optimizer, metrics = ['mse'])
        
        if load_weight_name != None :
            Q.load_weights(load_weight_name)
        if use_target_Q :
            Q_target = QNet(cfg.model_input_shape, cfg.actions_num)
            Q_target.set_weights(Q.get_weights())
        else :
            Q_target = None
        
        stepQueue = StepQueue()
        if cfg.use_reward == 0 :
            rewardFunc = DiffReward()
            curMapFunc = MapReward()
        elif cfg.use_reward == 1 :
            rewardFunc = TieredDiffReward()
            curMapFunc = MapReward()
        elif cfg.use_reward == 2 :
            rewardFunc = MapReward()
        
        loss_list = []
        avgR_list = []
        avgQ_list = []
        endMap_list = []
        testEndMap_list = []
        
        self.writeLog("epoch", "loss", "avrgR", "avrgQ", "endMap")
        
        for e in range(cfg.episodes) :
            
            self.game.newgame()
            sys.stdout.write(str(e) + ":")
            sys.stdout.flush()
            
            nxt_shot = self.game.getScreenshot(wait_no_move = False)
            
            this_epoch_eps = max(cfg.eps_min, cfg.epsilon * (cfg.eps_decay ** e))
            avgR_list.append(0)
            avgQ_list.append(0)
            loss_list.append(0)
            stuck_count = 0

            for n in range(cfg.steps_episode) :
                if (n + 1) % (cfg.steps_episode / 10) == 0 :
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                cur_shot = nxt_shot

                # make action
                predQ = np.squeeze(Q.predict(self.addNoise(cur_shot)))
                avgQ_list[e] += np.max(predQ) / cfg.steps_episode
                if np.random.random() < this_epoch_eps :
                    cur_action = np.random.randint(cfg.actions_num)
                else :
                    # soft max
                    w = predQ
                    w = np.exp(w)
                    w /= w.sum()
                    cur_action = np.random.choice(np.arange(cfg.actions_num), p = w)
                
                self.game.doControl(cur_action)
                
                nxt_shot = self.game.getScreenshot(wait_no_move = True)
                cur_reward = rewardFunc.getReward(cur_shot, nxt_shot) # pre-action, after-action
                stepQueue.addStep(cur_shot, cur_action, cur_reward)
                avgR_list[e] += cur_reward / cfg.steps_episode
                #print(cur_action, ",", cur_reward)
                
                # check if stuck
                if cfg.check_stuck :
                    if cur_reward == 0 :
                        stuck_count += 1 
                    elif stuck_count > 0 :
                        stuck_count -= 1
                    if stuck_count > cfg.stuck_thrshld :
                        loss_list[e] *= (float(cfg.steps_episode) / float(n))
                        avgR_list[e] *= (float(cfg.steps_episode) / float(n))
                        sys.stdout.write(str(n))
                        sys.stdout.flush()
                        break
                
                if stepQueue.getLength() > cfg.train_thrshld and n % cfg.steps_train == 0 :
                    # Experience Replay
                    random_step = np.random.randint(stepQueue.getLength() - (cfg.train_size + 1))
                    trn_s, trn_a, trn_r = stepQueue.getStepsAsArray(random_step, cfg.train_size + 1)
                    
                    # make next predicted reward array and train input array at same time
                    new_r = np.zeros((cfg.train_size, cfg.actions_num))
                    for j in range(cfg.train_size) :
                        if use_target_Q :
                            new_r[j] = Q_target.predict(np.expand_dims(trn_s[j], axis = 0))
                            predict_Q = Q_target.predict(np.expand_dims(trn_s[j+1], axis = 0))
                        else :
                            new_r[j] = Q.predict(np.expand_dims(trn_s[j], axis = 0))
                            predict_Q = Q.predict(np.expand_dims(trn_s[j+1], axis = 0))
                        new_r[j, trn_a[j]] = trn_r[j] + np.max(predict_Q) * cfg.gamma
                        # Q_new = r + Q_predict(a,s) * gamma
                    
                    #print("new_r\n", new_r[0])
                    
                    loss_list[e] += Q.train_on_batch(trn_s[:cfg.train_size], new_r)[0] / cfg.steps_episode * cfg.steps_train
                    
            # end for(STEP_PER_EPOCH)
            self.game.quitgame()
            
            # write log file and log list
            
            if cfg.use_reward == 0 :
                rewardFunc.clear() # DiffReward has to clear memory
                endMap = curMapFunc.getCurMap(cur_shot)
            elif cfg.use_reward == 1 :
                endMap = curMapFunc.getCurMap(cur_shot)
            elif cfg.use_reward == 2 :
                endMap = rewardFunc.getCurMap(cur_shot)
            endMap_list.append(endMap)
            print("loss: %.4f avrgQ: %.3f avrgR: %.3f end map %d" % (loss_list[e], avgQ_list[e], avgR_list[e], endMap))
            self.writeLog(str(e), str(loss_list[e]), str(avgQ_list[e]), str(avgR_list[e]), str(endMap))
            
            if use_target_Q and stepQueue.getLength() > cfg.train_thrshld :
                Q_target.set_weights(Q.get_weights())
                Q_target.save(save_weight_name)
            else :
                Q.save(save_weight_name)
                
            if (e + 1) % cfg.test_intv == 0 :
                test_endMap = self.test(save_weight_name, verdict = False)[0]
                testEndMap_list.append(test_endMap)
                print("test: ", test_endMap)
            
            if (e + 1) % cfg.draw_fig_intv == 0 :
                self.drawFig(loss_list, avgR_list, avgQ_list, endMap_list, testEndMap_list)
            
        # end for(episodes)
        self.drawFig(loss_list, avgR_list, avgQ_list, endMap_list, testEndMap_list)   
    # end def fit
    
    def random_action(self, steps = cfg.steps_test) :
        # click "NEW GAME"
        self.game.newgame()
        
        if cfg.use_reward == 0 :
            rewardFunc = DiffReward()
        elif cfg.use_reward == 1 :
            rewardFunc = TieredDiffReward()
        elif cfg.use_reward == 2 :
            rewardFunc = MapReward()
        
        for n in range(steps) :
            cur_shot = self.game.getScreenshot()
            cur_action = np.random.randint(cfg.actions_num)
            self.game.doControl(cur_action)
            nxt_shot = self.game.getScreenshot()
            cur_reward = rewardFunc.getReward(cur_shot, nxt_shot)
            print(cur_action, ",", cur_reward)
        
        print("test end, of reward: %.2f" % cur_reward)
        # Exit Game...
        self.game.quitgame()
    # end def
    
# end class DQN
    
    