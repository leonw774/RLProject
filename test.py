import sys
import train

print("how many eval will ran:")
c = int(input())

t = train.Train()
t.count_down(3)
#t.random_action()
model_name = sys.argv[1] if len(sys.argv) == 2 else "Q_model.h5"
print(t.test(model_name, c))