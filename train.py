import drl
from datetime import datetime, timedelta
from configure import Configuration as cfg 

if __name__ == '__main__' :
    my_drl = drl.DRL()
    my_drl.countdown(5)
    starttime = datetime.now()
    #my_drl.random_action(steps = 40)
    my_drl.fit(use_target_model = cfg.use_target_model)
    print(datetime.now() - starttime)
    #print(my_drl.test("Q_model.h5", rounds = 50))
    
    