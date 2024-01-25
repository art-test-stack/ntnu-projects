import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MLP:
    def __init__(
            self, 
            dim: tuple(int) = (30, 20, 40), 
            a_f: tuple(function) = (sigmoid, sigmoid, sigmoid),
            lr: float = .2
        ):
        pass
        

    def init_weights(self, dim):
        self.W = [ np.random.rand(dim[k]) for k in range(len(dim)) ]
        self.b = [ np.random.rand(1) for k in range(len(dim))]

    def split_minibatch(self):
        pass
    
    def forward(self, batch, size_minibatch: float = .8):
        pass
        