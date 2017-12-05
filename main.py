import numpy as np
import matplotlib.pyplot as plt
from esn import ESN, read_file, find_ind
import os
import pickle

data_dir = "./output/"
output_dir = "./nets/"
filenames = os.listdir(data_dir)
filenames.sort()
hidden_size = 3
input_size = 11
output_size = 2
steer_out = 1

file_ind = find_ind("forza_1", filenames)
for index in range(0, len(filenames), 3):

    esn = ESN(input_size, output_size, hidden_size)
    for file_index in range(index, index + 3):

        input_train, target_train, N_max = read_file(os.path.join(data_dir, filenames[file_index]), steer_out)

        epochs = 3

        #train
        for epoch in range(epochs):
            for ind in range(N_max - 1):
                inp = input_train[ind].reshape(-1, 1)
                desired_target = target_train[ind + 1].reshape(-1, 1)

                if ind >= esn.washout:
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


    #esn.store_net("forza.pickle")
    out_name = os.path.join(output_dir, filenames[index])
    esn.save_genome("%s.params" % out_name)


#for i in range(len(filenames)):
input_test, target_test, N_max_t = read_file(os.path.join(data_dir, filenames[1]), steer_out)

#test
m = 1800
start = 0
Y = np.zeros((N_max_t, output_size))
inp = input_test[0].reshape(-1, 1)
for ind in range(start + m):
    Y[ind] = esn.activate(inp).squeeze()
    inp = input_test[ind + 1].reshape(-1, 1)
    mse = esn.mse(Y[ind], target_test[ind + 1])
    print(ind)
    print("output", Y[ind].T)
    print("target", target_test[ind + 1])
    print("mse", mse)
    print("\n")

#plot

titles = ["speed", "position"]
for index in range(1, 3):
    plt.figure(index).clear()
    plt.plot(range(m), target_test[start + 1: start + m + 1, index - 1], label="target")
    plt.plot(range(m), Y[start:start + m, index - 1], label="predict")
    plt.legend()
    plt.title(titles[index - 1])

plt.show()



