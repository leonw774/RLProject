import sys
import train

train = train.Train()
train.count_down(3)
if sys.argv[1] != None :
    train.eval(sys.argv[1])
else :
    train.eval("Q_target_weight.h5")