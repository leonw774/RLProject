import sys
import dqn

print("how many episodes this test will ran:")
c = 100 #int(input())
t = dqn.DQN()
t.count_down(3)
#t.random_action()

model_name = sys.argv[1] if len(sys.argv) == 2 else "Q_model.h5"
this_goal = 11

# USE NO TRACKPAD TUNING

r = t.test("Q_model-12-17.h5", rounds = c, goal = 11, verdict = False)
resultfile = open("test_result-no-trackpad-11.csv", 'w')
if (r.shape == c) :
    resultfile.write("round, end_map\n")
    for i in range(c) :
        resultfile.write(str(i) + "," + str(r[i]) + "\n")
else :
    resultfile.write("round, step, goal\n")
    for i in range(c) :
        resultfile.write(str(i)  + "," + str(r[i, 0])  + ","  + str(r[i, 1]) + "\n")
 
######## GOAL 35 ########

r = t.test("Q_model-12-17.h5", rounds = c, goal = 35, verdict = False)
resultfile = open("test_result-no-trackpad-35.csv", 'w')
if (r.shape == c) :
    resultfile.write("round, end_map\n")
    for i in range(c) :
        resultfile.write(str(i) + "," + str(r[i]) + "\n")
else :
    resultfile.write("round, step, goal\n")
    for i in range(c) :
        resultfile.write(str(i)  + "," + str(r[i, 0])  + ","  + str(r[i, 1]) + "\n")