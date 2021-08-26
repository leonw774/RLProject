import drl
from datetime import datetime, timedelta

if __name__ == '__main__' :
    my_drl = drl.DRL()
    my_drl.countdown(3)
    starttime = datetime.now()
    #my_drl.random_action(steps = 40)
    my_drl.fit()
    print(datetime.now() - starttime)
    #print(my_drl.test("Q_model.h5", rounds = 50))
    
    