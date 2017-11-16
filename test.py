import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import functions as fun

np.random.seed(42)


output_size = 3
lamb = 0.001

#N_data = 400
len_interval = 16

#train data
#x, target = fun.gen_cosine(N_data, len_interval)


filename = ["./train_data/aalborg.csv", "./train_data/alpine-1.csv", "./train_data/f-speedway.csv"]
input_train, target_train, input_test, target_test, N, N_max, input_size = fun.read_file(filename[0], output_size)
N_data = N
hidden_sizes = [300]
washouts = [50]
sp_rads = [0.5, 0.9]



ind = 1
for hidden_size in hidden_sizes:
    for washout in washouts:
        for sp_rad in sp_rads:

            #init weigths
            W_in, W_bw, W = fun.init_weights(hidden_size, sp_rad, input_size, output_size)

            X = np.zeros((hidden_size + input_size, N_data - washout))
            Yt = target_train[washout:].T
            hidden = np.zeros((hidden_size))

            for t in range(N_data):
                inp = input_train[t]
                targ = target_train[t]
                hidden = fun.forward_pass(inp, targ, hidden, W_in, W, W_bw)

                if t >= washout:
                    X[:, t - washout] = np.concatenate((hidden.flatten(), inp.flatten()))

            W_out = fun.ridge(X, Yt, lamb, hidden_size, input_size)

            predictions = np.zeros((N, output_size))
            for t in range(N_data):
                inp = input_train[t]
                targ = target_train[t]
                out = fun.complete_for_pass(inp, targ, hidden, W_in, W, W_bw, W_out)
                mse = fun.mse(out, targ)
                predictions[t] = out

            plt.subplot(len(hidden_sizes), len(washouts)*len(sp_rads), ind)

            print(len(predictions))
            print(predictions[0])
            plt.plot(range(N_data-washout), target_train[washout:,2])
            plt.plot(range(N_data-washout), predictions[washout:, 2])
            plt.title(str(hidden_size) + " "+str(washout)+" "+str(sp_rad))

            ind += 1

plt.show()