import numpy as np
import matplotlib.pyplot as plt
from esn import ESN


esn = ESN(22, 4, 30)

#esn.store_net()
ran_index = 2000
ran_len = 200
m = ran_index - ran_len


filename = ["./train_data/aalborg.csv", "./train_data/alpine-1.csv", "./train_data/f-speedway.csv"]
input_train, target_train, N_max, input_size = esn.read_file(filename[0], 3)

Y = np.zeros((esn.output_size, m))
inp = input_train[:, ran_index].reshape(-1, 1)

for t in range(m):
    Y[:, t] = esn.activate(inp, sigm=False).squeeze()
    inp = input_train[:, ran_index + t + 1].reshape(-1, 1)

mse = esn.mse(Y, target_train[:, ran_index + 1: ran_index + m + 1])/m
print(mse)

titles = ["acceleration", "brake", "steer left", "steer right"]
for index in range(1, 5):
    plt.figure(index).clear()
    plt.plot(range(m), target_train[index - 1, ran_index + 1: ran_index + m + 1], label="target")
    plt.plot(range(m), Y[index - 1, :m], label="predict")
    plt.legend()
    plt.title(titles[index-1])

plt.show()