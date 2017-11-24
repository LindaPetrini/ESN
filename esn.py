import numpy as np
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ESN:

    def __init__(self, input_size, output_size, hidden_size=30, lamb=0.1, a=0.3, sp_rad=0.03, washout=100):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.washout = washout
        self.lamb = lamb
        self.a = a
        self.sp_rad = sp_rad

        self.hidden = np.zeros((self.hidden_size, 1))
        self.output = np.zeros((self.output_size, 1))

        self.W_in = (np.random.rand(self.hidden_size, 1 + self.input_size) - 0.5) * 1
        self.W = self.init_W()
        self.W_out = np.zeros((self.hidden_size, self.output_size))
        self.train()

    def activate(self, inp, sigm=False, tanh=False):
        self.forward_pass(inp)
        self.output = np.dot(self.W_out, np.vstack((1, inp, self.hidden)))
        if sigm:
            self.output[:2] = sigmoid(self.output[:2])
        if tanh:
            self.output[:2] = np.tanh(self.output[:2])
        return self.output

    def store_net(self, filename="net.pickle"):
        pickle.dump(self, open(filename, 'wb'))

    def reset(self):
        self.hidden = np.zeros((self.hidden_size, 1))
        self.output = np.zeros((self.output_size, 1))

    def init_W(self):
        W = np.random.rand(self.hidden_size, self.hidden_size) - 0.5
        sp_rad_risc = max(abs(np.linalg.eig(W)[0]))
        W *= (1 / sp_rad_risc)
        W *= self.sp_rad
        return W

    def train(self, ind=0):
        filenames = ["./train_data/aalborg.csv", "./train_data/alpine-1.csv", "./train_data/f-speedway.csv"]
        input_train, target_train, N, self.input_size = self.read_file(filenames[ind], 3)

        X = np.zeros((1 + self.hidden_size + self.input_size, N - 1 - self.washout))
        Yt = target_train[:, self.washout + 1:]

        for t in range(N - 1):
            inp = input_train[:, t].reshape(-1, 1)
            self.forward_pass(inp)
            if t >= self.washout:
                X[:, t - self.washout] = np.vstack((1, inp, self.hidden)).squeeze()

        self.W_out = self.ridge(X, Yt)

    def ridge(self, X, Yt):
        X_t = X.T
        W_out = np.dot(np.dot(Yt, X_t), np.linalg.inv(np.dot(X, X_t) + self.lamb * np.eye(1 + self.input_size + self.hidden_size)))
        return W_out

    def forward_pass(self, inp):
        self.hidden = (1 - self.a) * self.hidden + self.a * np.tanh(np.dot(self.W_in, np.vstack((inp, 1))) + np.dot(self.W, self.hidden))

    def read_file(self, filename, output_size):
        f = open(filename, 'r')
        lines = f.readlines()
        N_max = len(lines) - 2
        input_size = len(lines[0].split(",")) - 2

        input_train = np.zeros((input_size, N_max))
        target_train = np.zeros((output_size + 1, N_max))

        for ind, line in enumerate(lines[1:N_max]):

            target_train[:2, ind] = line.split(",")[:2]

            steer = float(line.split(",")[2])
            if steer >= 0:
                target_train[2, ind] = steer
                target_train[3, ind] = 0
            else:
                target_train[2, ind] = 0
                target_train[3, ind] = (-1) * steer

            input_train[:, ind] = line.split(",")[output_size:]

        # normalize
        input_train[0] = input_train[0] / np.amax(np.fabs(input_train[0]))
        input_train[3:] = input_train[3:] / np.amax(np.fabs(input_train[3:]))

        return input_train, target_train, N_max, input_size

    def mse(self, out, targ):
        return 0.5 * (np.sum((out - targ) ** 2))
