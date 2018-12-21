import sys
import train

print("how many episodes this test will ran:")
c = int(input())

t = train.Train()
t.count_down(3)
#t.random_action()
model_name = sys.argv[1] if len(sys.argv) == 2 else "Q_model.h5"
r = t.test(model_name, rounds = c, verdict = False)
print("final tree-passing rate:", sum(int(x >= 11) for x in r) / c)