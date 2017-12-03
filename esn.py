import numpy as np
import pickle
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x):
    return - np.log(1 / x - 1)


class ESN:
    def __init__(self, input_size, output_size, hidden_size=30, lamb=1, a=0.7, sp_rad=0.8,
                 washout=100, delta=0.1, online_training=True, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.washout = washout
        self.lamb = lamb
        self.a = a
        self.sp_rad = sp_rad
        self.delta = delta

        self.hidden = np.zeros((self.hidden_size, 1))
        self.output = np.zeros((self.output_size, 1))

        self.bias = bias

        if self.bias:
            self.W_in = (np.random.rand(self.hidden_size, 1 + self.input_size) - 0.5) * 1
            self.W = self.init_W()
            self.W_out = np.zeros((self.output_size, 1 + self.input_size + self.hidden_size))

            # stuff for online training
            if online_training:
                self.epsilon = 0
                self.g = 0
                self.P = 1 / self.delta * np.eye((1 + self.input_size + self.hidden_size))
                self.k = np.zeros((1, 1 + self.input_size + self.hidden_size))

            else:
                self.offline_train()

        else:
            self.W_in = (np.random.rand(self.hidden_size, self.input_size) - 0.5) * 1
            self.W = self.init_W()
            self.W_out = np.zeros((self.output_size, self.input_size + self.hidden_size))

            # stuff for online training
            if online_training:
                self.epsilon = 0
                self.g = 0
                self.P = 1 / self.delta * np.eye((self.input_size + self.hidden_size))
                self.k = np.zeros((1, self.input_size + self.hidden_size))

            else:
                self.offline_train()

    def forward_pass(self, inp):
        concats = np.array([1])
        concats = np.append(concats, inp).reshape(-1, 1)
        if self.bias:

            self.hidden = (1 - self.a) * self.hidden + self.a * np.tanh(np.dot(self.W_in, concats)
                                                                            + np.dot(self.W, self.hidden))

        else:
            self.hidden = (1 - self.a) * self.hidden + self.a * np.tanh(np.dot(self.W_in, inp)
                                                                        + np.dot(self.W, self.hidden))

    def activate(self, inp):
        self.forward_pass(inp)

        if self.bias:
            concats = np.array([1])
            concats = np.append(concats, inp)
            concats = np.append(concats, self.hidden).reshape(-1, 1)
            self.output = np.dot(self.W_out, concats)
        else:
            concats = np.append(inp, self.hidden).reshape(-1, 1)
            self.output = np.dot(self.W_out, concats)

        # self.output[2:] = sigmoid(self.output[2:])
        #
        # if self.output_size == 1:
        #     self.output[2] = np.tanh(self.output[2])
        # else:
        #     self.output[2:] = sigmoid(self.output[2:])

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

    def online_train(self, inp, target):
        self.output = self.activate(inp).reshape(-1, 1)

        targ = np.copy(target)

        # for ind in range(len(targ)):
        #     if abs(targ[ind]) == 1:
        #         targ[ind] -= targ[ind] * 5e-15
        #
        #     if targ[ind] == -1:
        #         print("AHHH")
        #
        #     if abs(targ[ind]) == 0:
        #         targ[ind] += 5e-15
        #
        # targ[2] = np.arctanh(targ[2])
        # targ[:2] = logit(targ[:2])

        self.epsilon = (targ - self.output)
        if self.bias:
            h = np.vstack([1, inp, self.hidden])
        else:
            h = np.vstack([inp, self.hidden])

        self.g = np.dot(self.P, h) / (self.lamb + np.dot(np.dot(h.T, self.P), h))

        self.P = 1 / self.lamb * (self.P - np.dot(np.dot(self.g, h.T), self.P))

        self.W_out += np.dot(self.g, self.epsilon.reshape((1, -1))).T

        return self.output

    def offline_train(self, ind=0):
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
        W_out = np.dot(np.dot(Yt, X_t),
                       np.linalg.inv(np.dot(X, X_t) + self.lamb * np.eye(1 + self.input_size + self.hidden_size)))
        return W_out

    def mse(self, out, targ):
        return 0.5 * (np.sum((out - targ) ** 2))

    def save_genome(self, file, parameters):
        np.savetxt(file, parameters.reshape(-1), header=str(self.input_size) + ', ' + str(self.output_size)
                                                        + ', ' + str(self.hidden_size))


def read_file(filename, ster_out=1, reduce=True, sizes=False):
    '''
    accel_0,brake_0,gear_0,gear2_0,steer_0,clutch_0,curTime_0,angle_0,curLapTime_0,damage_0,distFromStart_0,
    distRaced_0,fuel_0,lastLapTime_0,racePos_0,opponents_0,opponents_1,opponents_2,opponents_3,opponents_4,
    opponents_5,opponents_6,opponents_7,opponents_8,opponents_9,opponents_10,opponents_11,opponents_12,
    opponents_13,opponents_14,opponents_15,opponents_16,opponents_17,opponents_18,opponents_19,opponents_20,
    opponents_21,opponents_22,opponents_23,opponents_24,opponents_25,opponents_26,opponents_27,opponents_28,
    opponents_29,opponents_30,opponents_31,opponents_32,opponents_33,opponents_34,opponents_35,
    rpm_0,speedX_0,speedY_0,speedZ_0,track_0,track_1,track_2,track_3,track_4,track_5,track_6,track_7,
    track_8,track_9,track_10,track_11,track_12,track_13,track_14,track_15,track_16,track_17,track_18,
    trackPos_0,wheelSpinVel_0,wheelSpinVel_1,wheelSpinVel_2,wheelSpinVel_3,z_0,
    focus_0,focus_1,focus_2,focus_3,focus_4
    :param filename:
    :param output_size:
    :return:
    '''
    f = open(filename, 'r')
    lines = f.readlines()
    N_max = len(lines) - 2
    inputs = lines[0].split(",")

    inputs_to_find = ["angle_0", "speedX_0", "speedY_0", "track_0", "track_18", "trackPos_0"]
    targets_to_find = ["accel_0", "brake_0", "steer_0"]
    indexes_inp = {}
    indexes_out = {}

    for string in inputs_to_find:
        indexes_inp[string] = find_ind(string, inputs)

    for string in targets_to_find:
        indexes_out[string] = find_ind(string, inputs)

    if ster_out == 1:
        output_size = 3
    else:
        output_size = 4

    input_size = find_input_len(lines[1].split(","), indexes_inp, reduce)

    input_train = np.zeros((N_max, input_size))
    target_train = np.zeros((N_max, output_size))

    for ind, line in enumerate(lines[1:N_max]):
        lis = list(line.split(","))

        input_train[ind] = np.array(inp_to_array(lis, indexes_inp, reduce))
        target_train[ind] = np.array(target_to_array(lis, indexes_out, ster_out))

    # for i in range(len(target_train)):
    #    target_train[i] /= np.amax(np.fabs(target_train[i]))

    if sizes:
        return input_train, target_train, N_max, input_size, output_size
    else:
        return input_train, target_train, N_max

def find_ind(string, lis):
    index = None
    for i in range(len(lis)):
        if lis[i] == string:
            index = i
    return index


def find_input_len(lis, indexes, reduce=True):
    inputs = inp_to_array(lis, indexes, reduce)

    return len(inputs)


def inp_to_array(lis, indexes, reduce=True):
    '''
    inputs_to_find = ["angle_0", "speedX_0", "speedY_0", "track_0", "track_18", "trackPos_0"]
    :param list:
    :param indexes:
    :param reduce:
    :return:
    '''

    inp = list()
    inp.append(float(lis[indexes["angle_0"]]) / 180.0)
    inp.append(float(lis[indexes["speedX_0"]]) / 50.0)
    inp.append(float(lis[indexes["speedY_0"]]) / 40.0)

    distances_from_edge = lis[indexes["track_0"]:indexes["track_18"] + 1]
    distance_from_center = float(lis[indexes["trackPos_0"]])

    if reduce:
        for idxs in [[0], [2], [4], [7, 9, 11], [14], [16], [18]]:
            d = min([float(distances_from_edge[j]) for j in idxs])
            if math.fabs(distance_from_center) > 1 or d < 0:
                inp.append(-1)
            else:
                inp.append(d / 200.0)

    else:
        for element in distances_from_edge:
            inp.append(element)

    inp.append(distance_from_center)

    return inp


def target_to_array(lis, indexes, ster_out):
    '''
    ["accel_0", "brake_0", "steer_0"]
    :param list:
    :param indexes:
    :param ster_out:
    :return:
    '''
    target = list()
    target.append(lis[indexes["accel_0"]])
    target.append(lis[indexes["brake_0"]])

    if ster_out == 1:
        target.append(lis[indexes["steer_0"]])
    else:
        steer = float(lis[indexes["steer_0"]])
        if steer >= 0:
            target.append(steer)
            target.append(0.0)
        else:
            target.append(0.0)
            target.append((-1.0) * steer)

    return target
