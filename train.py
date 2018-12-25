import dqn
from datetime import datetime, timedelta
from setting import Setting as set 

if __name__ == '__main__' :
    my_dqn = dqn.DQN(use_target_Q = set.use_target_Q)
    my_dqn.count_down(5)
    starttime = datetime.now()
    #train.random_action()
    my_dqn.fit()
    print(datetime.now() - starttime)
    print(my_dqn.test("Q_model.h5", rounds = 50))
    
    