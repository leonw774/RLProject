import sys
import train

print("how many episodes this test will ran:")
c = int(input())
t = train.Train()
t.count_down(3)
#t.random_action()

model_name = sys.argv[1] if len(sys.argv) == 2 else "Q_model.h5"
this_goal = None

r = t.test(model_name, rounds = 5, max_step = 50, goal = this_goal, verdict = False)

resultfile = open("test_result.csv", 'w')
if (r.shape == c) :
    resultfile.write("round, end_map\n")
    for i in range(c) :
        resultfile.write(str(i) + "," + str(r[i]) + "\n")
else :
    resultfile.write("round, step, goal\n")
    for i in range(c) :
        resultfile.write(str(i)  + "," + str(r[i, 0])  + ","  + str(r[i, 1]) + "\n")