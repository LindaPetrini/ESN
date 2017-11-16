import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import functions as fun

np.random.seed(42)


input_size = 1
output_size = 1
lamb = 0.001

N_data = 400
len_interval = 16

#train data
x, target = fun.gen_cosine(N_data, len_interval)


hidden_sizes = [300]
washouts = [50]
sp_rads = [0.5, 0.9]



ind = 1
for hidden_size in hidden_sizes:
    for washout in washouts:
        for sp_rad in sp_rads:

            #init weigths
            W_in, W_bw, W = fun.init_weights(hidden_size, sp_rad)

            X = np.zeros((hidden_size + input_size, N_data - washout))
            Yt = target[washout:].T
            hidden = np.zeros((hidden_size))

            for t in range(N_data):
                inp = x[t]
                targ = target[t]
                hidden = fun.forward_pass(inp, targ, hidden, W_in, W, W_bw, vect=True)

                if t >= washout:
                    X[:, t - washout] = np.concatenate((hidden.flatten(), inp.flatten()))

            W_out = fun.ridge(X, Yt, lamb, hidden_size, input_size)

            predictions = []
            for t in range(N_data):
                inp = x[t]
                targ = target[t]
                out = fun.complete_for_pass(inp, targ, hidden, W_in, W, W_bw, W_out, vect=True)
                mse = fun.mse(out, targ)
                predictions.append(out)

            plt.subplot(len(hidden_sizes), len(washouts)*len(sp_rads), ind)
            plt.plot(x[washout:], target[washout:])
            plt.plot(x[washout:], predictions[washout:])
            plt.title(str(hidden_size)+ " "+str(washout)+" "+str(sp_rad))

            ind += 1

plt.show()