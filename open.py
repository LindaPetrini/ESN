import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim

class NET(nn.Module):

    def __init__(self, in_size, out_size, hidden_size,  parameters_file=None):
        super(NET, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(in_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h2o = nn.Linear(hidden_size + in_size, out_size, bias=False)
        self.sig = nn.Sigmoid()
        self.tan = nn.Tanh()


        a = np.random.binomial(1, 0.5, (hidden_size, hidden_size))
        b = np.random.normal(0, 1, (hidden_size, hidden_size))
        c = a*b
        sp_rad = max(np.absolute(np.linalg.eig(c)[0]))
        c *= 1/sp_rad
        c *= 0.8
        #sp_rad = max(np.fabs(np.linalg.eig(self.h2h.weight.data.numpy())[0]))

        self.h2h.weight.data = torch.Tensor(c)

        #print(np.linalg.eig(self.h2h.weight.data.numpy()))

        #max(abs(np.linalg.eig(self.h2h.weight.data.numpy())[0]))))

        #self.h2h.weight.data *= sp_rad.expand_as(self.h2h.weight)
        #self.h2h.weight.data.mul_(sp_rad)

        #if parameters_file is not None:

    def forward(self, input, hidden):
        a1 = self.i2h(input)
        a2 = self.h2h(hidden)
        hidden = self.sig(a1 + a2)
        output = self.h2o(torch.cat((hidden.squeeze(), input.squeeze())))
        output[:2] = self.sig(output[:2])
        output[2] = self.tan(output[2])

        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    #def save_parameters(self, file):
    #    np.savetxt(file, self.parameters())

torch.manual_seed(42)

f = open('./train_data/aalborg.csv', 'r')
#f = open('./train_data/alpine-1.csv', 'r')
#f = open('./train_data/f-speedway.csv', 'r')
lines = f.readlines()

input_size = len(lines[0].split(",")) - 3
output_size = 3
N_max = 400
#N_max = len(lines)-2
N = int(N_max * 0.8)
hidden_size = 8000  # size of reservoir
washout = 100

input_train = np.zeros((N, input_size))
target_train = np.zeros((N, output_size))

input_test = np.zeros((N_max - N, input_size))
target_test = np.zeros((N_max - N, output_size))

np.random.shuffle(lines[1:N_max])
for ind, line in enumerate(lines[1:N_max]):

    if ind < N:
        target_train[ind] = line.split(",")[:3]
        input_train[ind] = line.split(",")[3:-1]
    else:
        target_test[N-ind] = line.split(",")[:3]
        input_test[N-ind] = line.split(",")[3:-1]

model = NET(input_size, output_size, hidden_size)

#collect states
X = np.zeros((N-washout, input_size+hidden_size))
Yt = target_train[washout:N]

# run the reservoir with the data and collect X
hidden = model.init_hidden()
for t in range(N):
    net_input = Variable(torch.FloatTensor(input_train[t]))
    output, hidden = model.forward(net_input, hidden)
    hidden_nump = hidden.data.numpy()
    input_nump = net_input.data.numpy()
    if t >= washout:
        tmp = np.concatenate([hidden_nump.flatten(), input_nump.flatten()])
        X[t - washout, :] = np.concatenate([hidden_nump.flatten(), input_nump.flatten()])
        #print(tmp)


optimizer = optim.SGD(model.h2o.parameters(), lr= 0.01)
loss = nn.MSELoss()

for ITER in range(100):

    target = Variable(torch.FloatTensor(Yt))
    inp = Variable(torch.FloatTensor(X))
    #print(X)
    for_pass = model.h2o(inp)
    for_pass[:2] = model.sig(for_pass[:2])
    for_pass[2] = model.tan(for_pass[2])
    #print(for_pass)
    #print(target)
    error = loss(for_pass, target)

    model.h2o.zero_grad()
    error.backward()
    #model.h2o.weight.data.add_(-0.01 * model.h2o.weight.grad.data)
    optimizer.step()

    print("iter", ITER, "Err", error)




# train
#X_T = X.T
#pseud_S = np.linalg.inv(X_T @ X) @ X_T
#transposed = pseud_S @ Yt
#model.h2o.weigth = transposed.T



#print(target_test)
for i in range(N_max - N):
    test_input = Variable(torch.FloatTensor(input_test[i]))
    output, hidden = model.forward(test_input, hidden)
    target = Variable(torch.FloatTensor(target_test[i]))
    err = loss(output, target)
    #print(i, output, err)
