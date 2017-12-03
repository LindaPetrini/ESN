import numpy as np
import matplotlib.pyplot as plt
from esn import ESN, read_file
import os
import pickle

data_dir = "./output/"
filenames = os.listdir(data_dir)
filenames.sort()

hidden_size = 30
steer_out = 1

input_train, target_train, N_max, input_size, output_size = read_file(os.path.join(data_dir, filenames[0]), steer_out, sizes=True)

esn = ESN(input_size, output_size, hidden_size)

epochs = 1

#train
for epoch in range(epochs):
    for ind in range(N_max - 1):
        inp = input_train[ind].reshape(-1, 1)
        desired_target = target_train[ind + 1].reshape(-1, 1)

        if ind > esn.washout:
            out = esn.online_train(inp, desired_target)
            mse = esn.mse(out, desired_target)
            print(ind)
            print("target", desired_target.T)
            print("output", out.T)
            print("mse", mse)
            print("\n")

    esn.reset()

print("hid", esn.W)
print("in", esn.W_in)
print("out", esn.W_out)

#esn.store_net("aalborg.pickle")

#for i in range(len(filenames)):
input_test, target_test, N_max_t = read_file(os.path.join(data_dir, filenames[0]), steer_out)

#test
m = 1000
start = 500
Y = np.zeros((N_max_t, esn.output_size))
inp = input_test[0].reshape(-1, 1)
for ind in range(start + m):
    Y[ind] = esn.activate(inp).squeeze()
    inp = input_test[ind + 1].reshape(-1, 1)
    print(ind)
    print("output", Y[ind].T)
    print("target", target_test[ind + 1])
    print("\n")

#plot

if steer_out == 1:
    titles = ["acceleration", "brake", "steering"]
    for index in range(1, 4):
        plt.figure(index).clear()
        plt.plot(range(m), target_test[start + 1: start + m + 1, index - 1], label="target")
        plt.plot(range(m), Y[start: start + m,  index - 1], label="predict")
        plt.legend()
        plt.title(titles[index-1])
else:
    titles = ["acceleration", "brake", "steer left", "steer right"]
    for index in range(1, 5):
        plt.figure(index).clear()
        plt.plot(range(m), target_test[start + 1: start + m + 1, index - 1], label="target")
        plt.plot(range(m), Y[start :start + m, index - 1], label="predict")
        plt.legend()
        plt.title(titles[index - 1])

plt.show()



