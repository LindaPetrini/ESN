import numpy as np
import math
import matplotlib.pyplot as plt



#f = open('./train_data/aalborg.csv', 'r')
#f = open('./train_data/alpine-1.csv', 'r')
#f = open('./train_data/f-speedway.csv', 'r')
#lines = f.readlines()

def gen_cosine(n, num):
    x = np.linspace(0, num * math.pi, n)
    #t = np.random.normal(np.cos(x), 0.2)
    t = np.cos(x)

    return x, t

def init_stuff(hidden_size, sp_rad):
    W_in = np.random.normal(0, 1, (hidden_size, input_size))
    # W_out = np.random.normal(0, 1, (output_size, hidden_size + input_size + 1))
    W_bw = np.random.normal(0, 1, (hidden_size, output_size))
    b_in = np.random.normal(0, 1, (N_max, 1))
    b_out = np.random.normal(0, 1, (N_max, 1))

    # init W
    a = np.random.binomial(1, 0.1, (hidden_size, hidden_size))
    b = np.random.normal(0, 1, (hidden_size, hidden_size))
    W = a * b
    sp_rad_risc = np.amax(np.absolute(np.linalg.eig(W)[0]))
    W *= 1 / sp_rad_risc
    W *= sp_rad

    return W_in, W_bw, W

def forward_pass(hidden_size, washout):
    X = np.zeros((hidden_size + input_size, N - washout))
    Yt = target_train[washout:].T

    # init hidden
    hidden = np.zeros((hidden_size))

    for t in range(N):
        inp_t = input_train[t].T

        forz = target_train[t].T
        # tmp = W_in @ inp_t + W @ hidden + W_bw @ forz

        tmp = (W_in * inp_t).squeeze() + (W @ hidden.T) + (W_bw * forz).squeeze()

        hidden = sigmoid(tmp)

        if t >= washout:
            X[:, t - washout] = np.concatenate((hidden.flatten(), inp_t.flatten()))

    X_t = X.T

    return X, Yt






np.random.seed(42)
Nd = 1000
num = 6
input_train, target_train = gen_cosine(Nd, num)
N_max = Nd
N = Nd



#input_size = len(lines[0].split(",")) - 2
#output_size = 3
#N_max = len(lines)-2
#N = int(N_max * 0.8)

input_size = 1
output_size = 1
# hidden_size = 30  # size of reservoir
# washout = 10
# sp_rad = 0.5
lamb = 0.1

hidden_sizes = [50, 100, 300, 500]
washouts = [10, 100, 500]
sp_rads = [0.2, 0.5, 0.7]

for hidden_size in hidden_sizes:
    for washout in washouts:
        for sp_rad in sp_rads:



# input_train = np.zeros((N, input_size))
# target_train = np.zeros((N, output_size))
#
# input_test = np.zeros((N_max - N, input_size))
# target_test = np.zeros((N_max - N, output_size))
#
# print("reading data:")
# for ind, line in enumerate(lines[500:N_max]):
#     if ind < N:
#         target_train[ind] = line.split(",")[:3]
#         input_train[ind] = line.split(",")[3:]
#     else:
#         target_test[N-ind] = line.split(",")[:3]
#         input_test[N-ind] = line.split(",")[3:]
#     print("*", end="")
#
# input_train = input_train / np.amax(np.fabs(input_train))
# input_test /= np.amax(np.fabs(input_test))


#matrix init

            W_in, W_bw, W = init_stuff(hidden_size, sp_rad)



            #test
            # out = target_train[-1]
            # for t in range(N_max-N):
            #     inp_t = input_test[t].T
            #     tmp = W_in @ inp_t + W @ hidden + W_bw @ out
            #     hidden = sigmoid(tmp)
            #     out = W_out @ np.concatenate((hidden, inp_t))
            #     #out[:2] = sigmoid(out[:2])
            #     #out[2] = np.tanh(out[2])
            #
            #     mse = 0.5 * (np.sum((out - target_test[t]) ** 2))
            #     if target_test[t][0] == 0:
            #         print("iter: ", t, "mse:", mse)
            #         print("out", out)
            #         print("target", target_test[t], "\n")

            #test
            out = target_train[0]
            ls = []
            for t in range(N):
                inp_t = input_train[t].T

                #tmp = W_in @ inp_t + W @ hidden + W_bw @ out
                tmp = (W_in * inp_t).squeeze() + (W @ hidden) + (W_bw * out).squeeze()

                hidden = sigmoid(tmp)
                out = W_out @ np.concatenate((hidden.flatten(), inp_t.flatten()))
                #out[:2] = sigmoid(out[:2])
                #out[2] = np.tanh(out[2])

                mse = 0.5 * (np.sum((out - target_train[t]) ** 2))
                #if target_train[t][0] == 0:
                print("iter: ", t, "mse:", mse)
                print("out", out)
                print("target", target_train[t], "\n")
                ls.append(out)

            x = np.linspace(0, num * math.pi, Nd)
            plt.figure(ind)
            plt.plot(x, target_train)
            plt.plot(x, ls)
            plt.show()