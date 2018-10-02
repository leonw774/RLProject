import sys
import train

t = train.Train()
t.count_down(3)
t.random_action()
if len(sys.argv) == 2 :
    t.eval(sys.argv[1])
else :
    t.eval("Q_target_model.h5")