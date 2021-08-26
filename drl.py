import sys
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from keras.models import Model, load_model

from configure import Configuration as cfg
from mymodels import ActorCritic
from stepqueue import StepQueue
from gameagent import GameAgent, Action
import gamereward #getCurMap PixelDiffReward, TieredPixelDiffReward, MapReward

class DRL():
    
    def __init__(self):
        self.logfile = open("log.csv", 'w')
        self.game = GameAgent()
    
    def countdown(self, cd):
        for i in range(cd):
            print(cd - i)
            sleep(1.0)
    
    def addNoise(self, noisy_scrshot):
        noisy_scrshot += np.random.uniform(low = -cfg.noise_range, high = cfg.noise_range, size = noisy_scrshot.shape)
        noisy_scrshot[noisy_scrshot > 1.0] = 1.0
        noisy_scrshot[noisy_scrshot < 0.0] = 0.0
        return noisy_scrshot
    # end def get_screen_rect
    
    def write_log(self, values):
        log_list = []
        for v in values:
            if len(v) > 0 : log_list.append(str(v[-1]))
            else : log_list.append("na")
        log_string = ",".join(log_list)
        self.logfile.write(log_string + "\n")
    
    def drawFig(self, history):
        for key, value in history.items():
            if len(value) > 0:
                plt.figure(figsize = (12, 8))
                plt.xlabel("epoch")
                plt.plot(value, label = key)
                if key == "endmap" or key == "test_endmap":
                    the_tree = [11] * len(value)
                    plt.plot(the_tree, label = "The Tree")
                plt.legend(loc = "upper right")
                plt.savefig("fig/" + key + ".png")
                plt.close()
        
    def test(self, model_weight_name, rounds = 1, max_step = cfg.steps_test, verdict = False):
        testActor = load_model(model_weight_name)
        if verdict:
            print("Test begin for:", model_weight_name, "Step Limit:", max_step)    
        
        for i in range(rounds):
            if verdict : print("\nRound", i, "begin")
            # click "NEW GAME"
            self.game.newgame()
            orig_cur_shot, input_cur_shot = self.game.getScreenshot(wait_still = False)
            
            for n in range(max_step):
                cur_action = Action(np.squeeze(testActor.predict(input_cur_shot)))
                #print(action_Q)
                if verdict: print("Score:", gamereward.getCurMap(input_cur_shot), "\nDo action", cur_action, "with Q:", action_Q[cur_action], "\n")
                    
                self.game.doControl(cur_action)
                
                orig_cur_shot, input_cur_shot = self.game.getScreenshot(wait_still = True)
            # end for max_step
            
            endmap = gamereward.getCurMap(input_cur_shot)
            if verdict: print("Test round", i, "ended. endmap:", endmap)
            # exit game...
            self.game.quitgame()
            return endmap

    # end def test
    
    def fit(self, load_weight_name = None, save_weight_name = "new_model.h5"):

        stuck_count = 0
        def check_stuck(cur_reward, stuck_count):
            if cur_reward == 0:
                stuck_count += 1 
            elif stuck_count > 0:
                stuck_count -= 1
            return stuck_count > cfg.stuck_thrshld
        # end def check_stuck

        stepQueue = StepQueue()
        rewardFunc = getattr(gamereward, cfg.reward_func_name)()
        history = {"epoch" : [], "endmap" : [], "test_endmap" : [], "avg_reward" : []}
        
        myModel = ActorCritic(load_weight_name, cfg.model_input_shape)
        history["model_loss"] = []

        self.write_log(history.keys())
        
        for e in range(cfg.episodes):
            
            self.game.newgame()
            sys.stdout.write(str(e) + ":"); sys.stdout.flush()
            
            orig_nxt_shot, input_nxt_shot = self.game.getScreenshot(wait_still = False)
            cur_epsilon = max(cfg.epsilon_min, cfg.init_epsilon * (cfg.epsilon_decay ** e))
            stuck_count = 0
            
            history["avg_reward"].append(0)
            history["epoch"].append(e)
            history["model_loss"].append(list())
            
            for n in range(cfg.steps_per_episode):
                if (n + 1) % (cfg.steps_per_episode / 10) == 0:
                    sys.stdout.write("."); sys.stdout.flush()
                
                orig_cur_shot, input_cur_shot = orig_nxt_shot, input_nxt_shot

                # make action
                if np.random.random() > cur_epsilon:
                    cur_action = myModel.decision(input_cur_shot)
                else:
                    '''
                    time: 0 ~ 1 (second)
                    speed: 0 ~ 50
                    '''
                    cur_action = Action([np.random.random(), np.random.random() * 50, np.random.random(), np.random.random() * 2 - 1])
                
                self.game.doControl(cur_action)
                
                orig_nxt_shot, input_nxt_shot = self.game.getScreenshot(wait_still = True)
                cur_reward = rewardFunc.getReward(orig_cur_shot, orig_nxt_shot) # pre-action, after-action
                stepQueue.addStep(input_cur_shot, cur_action, cur_reward)
                history["avg_reward"][-1] += cur_reward / cfg.steps_per_episode
                #print(cur_action, ",", cur_reward)
                
                # check if stuck
                if cfg.check_stuck:
                    if (check_stuck(cur_reward, stuck_count)):
                        sys.stdout.write(str(n)); sys.stdout.flush()
                        break
                
                if stepQueue.getLength() > cfg.train_thrshld and n % cfg.steps_per_train == 0:
                    # Experience Replay
                    random_step = np.random.randint(stepQueue.getLength() - (cfg.train_size + 1))
                    trn_s, trn_a, trn_r = stepQueue.getStepsAsArray(random_step, cfg.train_size + 1)
                    loss = myModel.learn(trn_s, trn_a, trn_r)
                    history["model_loss"][-1].append(loss)

            # end for(STEP_PER_EPOCH)
            
            # write log file and log list
            rewardFunc.clear()
            endmap = gamereward.getCurMap(input_cur_shot)
            history["endmap"].append(endmap)
            history["model_loss"][-1] = tuple(np.mean(history["model_loss"][-1], axis=0))
            
            sys.stdout.write("\n")
            for key, value in history.items():
                if key != "test_endmap":
                    sys.stdout.write(key + " " + str(value[-1]) + "\n")
            sys.stdout.flush()
            self.write_log(history.values())
            
            # back to main menu
            self.game.quitgame()
            
            if stepQueue.getLength() > cfg.train_thrshld:
                myModel.save(save_weight_name)
                
            if (e + 1) % cfg.test_intv == 0:
                test_endmap = self.test(save_weight_name, verdict = False)
                history["test_endmap"].append(test_endmap)
                print("test: ", test_endmap)
            
            if (e + 1) % cfg.draw_fig_intv == 0:
                self.drawFig(history)
            
        # end for(episodes)
    # end def fit
    
# end class DQL
    
    