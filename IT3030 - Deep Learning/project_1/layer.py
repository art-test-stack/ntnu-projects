import numpy as np
from utils import HiddenLayerConfig
from functions import * # sigmoid, d_sigmoid, tanh, d_tanh, linear, d_linear, relu, d_relu, softmax, d_softmax, cross_entropy, d_cross_entropy, mse, d_mse

class Layer:

    def __init__(self, config: HiddenLayerConfig, lrate) -> None:
        
        for k, v in config.items():
            setattr(self, k, v)
        if 'lrate' not in config.keys():
            self.lrate = lrate

    def init_weights(self, size_input, w_range = (0, 1)):
        self.W = np.random.uniform(w_range[0], w_range[1], size=(size_input, self.size))
        self.b = np.random.uniform(w_range[0], w_range[1], size=(1, self.size))

    def forward_pass(self, input):
        self.hlast = input
        x = input.dot(self.W) + self.b
        self.h = act_function[self.act](x)
        return self.h

    def backward_pass(self, J):
        self.g = J * d_act_function[self.act](self.h)
        # self.dw = np.mean(np.einsum('ij,ik->ijk', self.hlast, self.g), axis=0)
        self.dw = (self.hlast).T.dot(self.g) / self.hlast.shape[0]
        self.db = np.mean(self.g, axis = 0)
        return self.g.dot(self.W.T)
    
    def update_weights(self):
        self.W += self.lrate * self.dw
        self.b += self.lrate * self.db