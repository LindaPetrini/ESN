import numpy as np
import math
import matplotlib.pyplot as plt




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gen_cosine(n, num, noise=False):
    x = np.linspace(0, num * math.pi, n)
    if noise:
        t = np.random.normal(np.cos(x), 0.2)
    else:
        t = np.cos(x)

    return x, t

def init_W(hidden_size, sp_rad, sparse=0.2):
    # init W
    a = np.random.binomial(1, sparse, (hidden_size, hidden_size))
    b = np.random.normal(0, 1, (hidden_size, hidden_size))
    W = a * b
    sp_rad_risc = np.amax(np.absolute(np.linalg.eig(W)[0]))
    W *= 1 / sp_rad_risc
    W *= sp_rad

    return W

def init_weights(hidden_size, sp_rad, input_size, output_size):
    W_in = np.random.normal(0, 1, (hidden_size, input_size))
    W_bw = np.random.normal(0, 1, (hidden_size, output_size))
    W = init_W(hidden_size, sp_rad)

    return W_in, W_bw, W

def forward_pass(inp, targ, hidden, W_in, W, W_bw, vect=False):
    if vect:
        tmp = (W_in * inp).squeeze() + (W @ hidden.T) + (W_bw * targ).squeeze()
    else:
        tmp = W_in @ inp + W @ hidden + W_bw @ targ
    hidden = sigmoid(tmp)

    return hidden

def complete_for_pass(inp, targ, hidden, W_in, W, W_bw, W_out, vect=False):

    hidden = forward_pass(inp, targ, hidden, W_in, W, W_bw, vect)
    out = W_out @ np.concatenate((hidden.flatten(), inp.flatten()))

    return out


def ridge(X, Yt, lamb, hidden_size, input_size):
    X_t = X.T
    mat = np.linalg.inv(X @ X_t + lamb * np.eye(hidden_size + input_size))
    W_out = Yt @ X_t @ mat
    return W_out

def mse(out, targ):
    return 0.5 * (np.sum((out - targ) ** 2))

def read_file(filename, output_size):

    f = open(filename, 'r')
    lines = f.readlines()
    N_max = len(lines)-2
    N = int(N_max * 0.8)
    input_size = len(lines[0].split(",")) - 2

    input_train = np.zeros((N, input_size))
    target_train = np.zeros((N, output_size))

    input_test = np.zeros((N_max - N, input_size))
    target_test = np.zeros((N_max - N, output_size))

    for ind, line in enumerate(lines[1:N_max]):
        if ind < N:
            target_train[ind] = line.split(",")[:output_size]
            input_train[ind] = line.split(",")[output_size:]
        else:
            target_test[N-ind] = line.split(",")[:output_size]
            input_test[N-ind] = line.split(",")[output_size:]

    input_train = input_train / np.amax(np.fabs(input_train))
    input_test /= np.amax(np.fabs(input_test))

    return input_train, target_train, input_test, target_test, N, N_max, input_size



