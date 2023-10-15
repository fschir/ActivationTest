import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, init: str):
        super(RNN, self).__init__()
        self.activation = {}

        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)
        # self.rnn.register_forward_hook(self.log_activation('LSTM'))
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(64, 32)
        self.fc_1.register_full_backward_hook(self.log_activation('Full_1'))
        self.fc_2 = nn.Linear(32, 10)
        self.fc_2.register_full_backward_hook(self.log_activation('Full_2'))

        self.layers = nn.Sequential(
            self.batchnorm,
            self.fc_1,
            self.fc_2,
        )

        if init == 'normal':
            for layer in self.layers:
                torch.nn.init.normal_(layer.weight, 0, 1)
                torch.nn.init.normal_(layer.bias, 0, 1)

        if init == 'kaiming':
            for layer in self.layers:
                try:
                    torch.nn.init.kaiming_normal_(layer.weight)
                except ValueError:
                    print(f'Layer: {layer} initialized normal')
                    torch.nn.init.normal_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)

    def log_activation(self, name):
        def hook(module, input, output):
            try:
                self.activation[name] = module.weight.grad
                # print(module.weight.grad)
            except AttributeError:
                self.activation[name] = output
        return hook

    def forward(self, inp):
        inp = inp.reshape(-1, 28, 28)
        output, hidden = self.rnn(inp)
        output = output[:, -1, :]
        output = self.batchnorm(output)
        # output = self.dropout_1(output)
        output = self.fc_1(output)
        output = F.relu(output)
        # output = self.dropout_2(output)
        output = self.fc_2(output)
        output = F.log_softmax(output, dim=1)
        return output


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        inp = x.reshape(-1, 28, 28)
        out, _ = [layer(inp) for layer in self.layers]
        return out
