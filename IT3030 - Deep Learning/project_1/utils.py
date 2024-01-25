import yaml
from enum import Enum


class Loss(Enum):
    cross_entropy: 'cross_entropy'
    mse: 'mse'

class Regularizator(Enum):
    L1: 'L1'
    L2: 'L2'
    none: None

class ActivationFunction(Enum):
    sigmoid: 'sigmoid'
    tanh: 'tanh'
    relu: 'relu'
    linear: 'linear'
    
class OutputActFunction(Enum):
    softmax: 'softmax'

class HiddenLayerConfig:
    size: int = 100
    act: str = 'relu'
    wr: tuple | str
    lrate: float = 0.01

class InputLayerConfig:
    input: int = 20

class OutputLayerConfig:
    type: OutputActFunction = 'relu'


class GlobalConfig:
    loss: Loss
    lrate: float = 0.1
    wreg: float = 0.001
    wrt: Regularizator

    def __init__(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

class LayersConfig:
    input: InputLayerConfig.input
    layers: list[HiddenLayerConfig]
    type: OutputLayerConfig.type

    def __init__(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def adapt_config(config):
    global_config = GlobalConfig(**config['GLOBAL'])
    layers_config = LayersConfig(**config['LAYERS'])
    return global_config, layers_config
