import drl
from datetime import datetime, timedelta
from configure import Configuration as cfg 

if __name__ == '__main__' :
    my_drl = drl.DRL()
    my_drl.countdown(5)
    starttime = datetime.now()
    #my_dqn.random_action(steps = 40)
    my_dqn.fit(use_target_Q = cfg.use_target_Q)
    print(datetime.now() - starttime)
    #print(my_dqn.test("Q_model.h5", rounds = 50))
    
    