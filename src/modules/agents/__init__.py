from .rnn_agent import RNNAgent
from .s_rnn_agent import SRNNAgent


REGISTRY = dict()
REGISTRY["rnn"] = RNNAgent
REGISTRY["s_rnn"] = SRNNAgent