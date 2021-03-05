import torch
import torch.nn as nn

mu_range = [(-0.3, 0.3), (-0.3, 0.3)]


class Network(nn.Module):
    def __init__(self, mu_len, c, N):
        nn.Module.__init__(self)
        self.c = c
        if self.c in ("u", "p"):
            input_size = mu_len
            hidden_size = 7
            output_size = N

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
        else:
            raise RuntimeError("Invalid component")

    def forward(self, x):
        if self.c in ("u", "p"):
            # [batch_size, number_of_unknown, 1] to [batch_size, number_of_unknown]
            x = x.view(x.shape[0], -1)

            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            return x
        else:
            raise RuntimeError("Invalid component")


class NormalizeInputs(object):
    def __init__(self, mu_range):
        min_ = [mu_range_p[0] for mu_range_p in mu_range]
        len_ = [mu_range_p[1] - mu_range_p[0] for mu_range_p in mu_range]
        self.min = torch.FloatTensor(min_).view(len(min_), -1)
        self.len = torch.FloatTensor(len_).view(len(len_), -1)

    def __call__(self, mu):
        if not isinstance(mu, torch.Tensor):
            assert isinstance(mu, tuple)
            mu = torch.FloatTensor([mu]).view(len(mu), -1)
            return mu.sub_(self.min).div_(self.len)
        else:
            for t in mu:
                t.sub_(self.min).div_(self.len)
            return mu


class NormalizeOutputs(object):
    def __init__(self, filename):
        with open(filename, "r") as infile:
            bounds = [line.rstrip("\n") for line in infile]
        self.min, self.max = bounds
        self.min, self.max = float(self.min), float(self.max)

    def __call__(self, y):
        return (y - self.min) / (self.max - self.min)

    def inv(self, y):
        return (self.max - self.min)*y + self.min
