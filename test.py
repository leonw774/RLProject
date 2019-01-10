import sys
import dqn

round_num = 30 #int(input())
t = dqn.DQN()

#model_name = sys.argv[1] if len(sys.argv) == 2 else "Q_model.h5"

# USE TRACKPAD TUNING
######## GOAL 35 ########
r = t.test("Q_model-12-15.h5", epsilon = 0.1, rounds = round_num, max_step = 50, goal = 34, verdict = True)

'''
resultfile = open("test_result.csv", 'w')
if (r.shape == c) :
    resultfile.write("round, end_map\n")
    for i in range(c) :
        resultfile.write(str(i) + "," + str(r[i]) + "\n")
else :
    resultfile.write("round, step, goal\n")
    for i in range(c) :
        resultfile.write(str(i)  + "," + str(r[i, 0])  + ","  + str(r[i, 1]) + "\n")
'''
 