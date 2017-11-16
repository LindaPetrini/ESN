import numpy as np
import math
import matplotlib.pyplot as plt

input_size = 1
output_size = 1



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

def init_weights(hidden_size, sp_rad):
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



