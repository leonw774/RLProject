import sys
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from keras.models import Model, load_model, optimizers

from configure import Configuration as cfg
from stepqueue import StepQueue
from qnet import QNet
from actorcritic import ActorCritic
from gameagent import GameAgent
from gamereward import DiffReward, TieredDiffReward, MapReward

class DRL() :
    
    def __init__(self) :
        self.logfile = open("log.csv", 'w')
        self.game = GameAgent()
            
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
    
    def write_log(self, values) :
        log_string = ','.join(values)
        self.logfile.write(log_string + "\n")
    
    def drawFig(self, history) :
        for key, value in history.items() :
            plt.figure(figsize = (12, 8))
            plt.xlabel("epoch")
            plt.plot(value, label = key)
            if key == "endmap" or key == "test_endmap" :
                the_tree = [11] * len(value)
                plt.plot(the_tree, label = "The Tree")
            plt.legend(loc = "upper right")
            plt.savefig("fig/" + key + ".png")
            plt.close()
        
    def test(self, model_weight_name, epsilon = cfg.eps_test, rounds = 1, max_step = cfg.steps_test, goal = None, verdict = False) :

        testModel = load_model(model_weight_name)
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
            cur_shot = self.game.getScreenshot(wait_still = True) 
            
            for n in range(max_step) :
                cur_action = Q.decision(self.addNoise(cur_shot), temperature = 0.1)
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
    
    def fit(self, load_weight_name = None, save_weight_name = "Q_model.h5", use_target_model = False) :
        '''
        We will train model, at time t, as:
            y_pred = model([state_t, a_t])
            y_true = r_t + gamma * max(Q_target([state_t, for a in A]))
        update Q's weight in mse
        after a number of steps, copy model to model_target.
        '''
        if cfg.use_model_name == "QNET" :
            myModel = QNet(use_target_model, load_weight_name, cfg.model_input_shape, cfg.actions_num)
        elif cfg.use_model_name == "AC" :
            myModel = ActorCritic(use_target_model, load_weight_name, cfg.model_input_shape, cfg.actions_num)
        
        stepQueue = StepQueue()
        if cfg.use_reward == 0 :
            rewardFunc = DiffReward()
            curMapFunc = MapReward()
        elif cfg.use_reward == 1 :
            rewardFunc = TieredDiffReward()
            curMapFunc = MapReward()
        elif cfg.use_reward == 2 :
            rewardFunc = MapReward()
        
        history = { "epoch" : [],
                    "endmap" : [],
                    "test_endmap" : [],
                    "avg_reward" : [],
                    "qnet_loss" : [],
                    "actor_loss" : [],
                    "critic_loss" : []
                  }
        self.write_log(history.keys())
        
        for e in range(cfg.episodes) :
            
            self.game.newgame()
            sys.stdout.write(str(e) + ":")
            sys.stdout.flush()
            
            nxt_shot = self.game.getScreenshot(wait_still = False)
            cur_epoch_eps = max(cfg.eps_min, cfg.epsilon * (cfg.eps_decay ** e))
            stuck_count = 0
            
            history["epoch"].append(e)
            history["avg_reward"].append(0)
            if cfg.use_model_name == "QNET" :
                history["qnet_loss"].append(0)
            elif cfg.use_model_name == "AC" :
                history["actor_loss"].append(0)
                history["critic_loss"].append(0)
            
            for n in range(cfg.steps_episode) :
                if (n + 1) % (cfg.steps_episode / 10) == 0 :
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                cur_shot = nxt_shot

                # make action
                if np.random.random() < cur_epoch_eps :
                    cur_action = np.random.randint(cfg.actions_num)
                else :
                    cur_action = myModel.decision(self.addNoise(cur_shot))
                
                self.game.doControl(cur_action)
                
                nxt_shot = self.game.getScreenshot(wait_still = True)
                cur_reward = rewardFunc.getReward(cur_shot, nxt_shot) # pre-action, after-action
                stepQueue.addStep(cur_shot, cur_action, cur_reward)
                history["avg_reward"] += cur_reward / cfg.steps_episode
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
                    
                    if cfg.use_model_name == "QNET" :
                        loss = myModel.learn(trn_s, trn_a, trn_r)
                        history["qnet_loss"][-1] += loss / cfg.steps_episode * cfg.steps_train
                    elif cfg.use_model_name == "AC" :
                        aloss, closs = myModel.learn(trn_s, trn_a, trn_r)
                        history["actor_loss"][-1] += aloss / cfg.steps_episode * cfg.steps_train
                        history["critic_loss"][-1] += closs / cfg.steps_episode * cfg.steps_train                    
                    
            # end for(STEP_PER_EPOCH)
            self.game.quitgame()
            
            # write log file and log list
            
            if cfg.use_reward == 0 :
                rewardFunc.clear() # DiffReward has to clear memory
                endmap = curMapFunc.getCurMap(cur_shot)
            elif cfg.use_reward == 1 :
                endmap = curMapFunc.getCurMap(cur_shot)
            elif cfg.use_reward == 2 :
                endmap = rewardFunc.getCurMap(cur_shot)
            history["endmap"].append(endmap)
            for key, value in history.items() :
                if key != "test_endmap" : sys.stdout.write(key + str(value[-1]) + " ")
            sys.stdout.flush()
            self.write_log(history.values())
            
            if stepQueue.getLength() > cfg.train_thrshld :
                myModel.save(save_weight_name)
                
            if (e + 1) % cfg.test_intv == 0 :
                test_endmap = self.test(save_weight_name, verdict = False)[0]
                history["test_endmap"].append(test_endmap)
                print("test: ", test_endMap)
            
            if (e + 1) % cfg.draw_fig_intv == 0 :
                self.drawFig(history)
            
        # end for(episodes)
        self.drawFig(loss_list, avgR_list, endmap_list, test_endmap_list)   
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
    
# end class DQL
    
    