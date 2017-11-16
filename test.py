import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import functions as fun

np.random.seed(42)


output_size = 3
lamb = 0.01

#N_data = 400
len_interval = 16

#train data
#x, target = fun.gen_cosine(N_data, len_interval)


filename = ["./train_data/aalborg.csv", "./train_data/alpine-1.csv", "./train_data/f-speedway.csv"]
input_train, target_train, input_test, target_test, N, N_max, input_size = fun.read_file(filename[2], output_size)
N_data = N
hidden_sizes = [30]
washouts = [50]
sp_rads = [0.2]



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
                hidden = fun.forward_pass(input_train[t], target_train[t], hidden, W_in, W, W_bw)
                if t >= washout:
                    X[:, t - washout] = np.concatenate((hidden.flatten(), input_train[t].flatten()))

            W_out = fun.ridge(X, Yt, lamb, hidden_size, input_size)

            #training data

            # predictions = np.zeros((N, output_size))
            # for t in range(N_data):
            #     inp = input_train[t]
            #     targ = target_train[t]
            #     out = fun.complete_for_pass(inp, targ, hidden, W_in, W, W_bw, W_out)
            #     mse = fun.mse(out, targ)
            #     predictions[t] = out
            #
            # plt.subplot(len(hidden_sizes), len(washouts)*len(sp_rads), ind)
            #
            # plt.plot(range(N_data-washout), target_train[washout:, 0])
            # plt.plot(range(N_data-washout), predictions[washout:, 0])
            # plt.title(str(hidden_size) + " "+str(washout)+" "+str(sp_rad))

            #test data
            predictions = np.zeros((N_max - N_data, output_size))
            total_err = 0
            for t in range(N_max - N_data):
                predictions[t] = fun.complete_for_pass(input_test[t], target_test[t], hidden, W_in, W, W_bw, W_out)
                mse = fun.mse(predictions[t], target_test[t])
                total_err += mse

                if target_test[t][0] == 0:
                    print("target: ", target_test[t])
                    print("prediction:", predictions[t])
                    print("mse: ", mse, "\n")

            total_err /= (N_max-N_data)

            plt.subplot(len(hidden_sizes), len(washouts) * len(sp_rads), ind)

            m = 200
            plt.plot(range(m), target_test[m:m+m, 2], label = "target")
            plt.plot(range(m), predictions[m:m+m, 2], label = "predict")
            plt.legend()
            plt.title(str(hidden_size) + " " + str(washout) + " " + str(sp_rad)+" "+ str(total_err))

            ind += 1

plt.show()