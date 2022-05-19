import torch.nn as nn
import torch.nn.functional as F


class SRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SRNNAgent, self).__init__()
        self.args = args

        self.fc_s = nn.Linear(input_shape['state'], args.rnn_hidden_dim)
        self.fc_o = nn.Linear(input_shape['obs'], args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_f = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc_s.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        b, a, e_s = inputs['state'].size()
        _, _, e_o = inputs['obs'].size()

        x_s = F.relu(self.fc_s(inputs['state'].view(-1, e_s)), inplace=True)
        x_o = F.relu(self.fc_o(inputs['obs'].view(-1, e_o)), inplace=True)
        x = None
        if self.args.operation == 'add':
            x = x_s + x_o
        elif self.args.operation == 'dot':
            x = x_s * x_o

        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        q = self.fc_f(h)
        return q.view(b, a, -1), h.view(b, a, -1)