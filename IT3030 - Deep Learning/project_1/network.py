import numpy as np
from utils import GlobalConfig, LayersConfig, read_config, adapt_config
from tqdm import tqdm

from layer import Layer
from matplotlib import pyplot as plt
from functions import * 



def has_nan(np_tab, name, ep=''):
    if np.isnan(np_tab).all():
        assert False, f'{name} has nan: {ep}'


def print_empty_tab_line(width=40):
    print(f"|{'-'* width}", end='|\n')

def print_tab_line(row1, row2 = '', width = 40):
    print(f"| {row1}{' '*(9 - len(row1) - 1)} | {row2}{' '*(width - len(row2) - 12)}", end='|\n')

def print_epoch_scores(epoch, loss, val_loss=None):
    epoch, loss = str(epoch), str(loss)
    val_loss_print = f" val loss: {str(val_loss)}" if val_loss else ""

    print(f"Epoch: {epoch},{' '*(5 - len(epoch))} loss: {loss},{' '*(18 - len(loss))}{val_loss_print}")


class Network:

    def __init__(self,
        global_config: GlobalConfig,
        layers_config: LayersConfig,
        show_config: bool = True) -> None:
        
        for k, v in global_config.__dict__.items():
            setattr(self, k, v)

        self.input = layers_config.input
        self.type = layers_config.type

        self.layers = []
        for c_layer in layers_config.hidden_layers:
            self.layers.append(Layer(c_layer, self.lrate))
            
        self.global_config = global_config
        self.layers_config = layers_config
        if show_config: 
            self.print_config()

    def forward_pass(self, X_train):
        h = X_train

        for layer in self.layers:
            h = layer.forward_pass(h)

        return type_function[self.type](h)
    
    def backward_pass(self, y_train, y_pred):
        dL = d_loss[self.loss](y_train, y_pred)
        dy = d_softmax(y_pred)

        # J = dL * dy
        # J = np.einsum('ij,ijk->ij', dL, dy)
        J = dL

        for layer in reversed(self.layers):
            J = layer.backward_pass(J)
    
    def init_weights(self): 
        input_size = self.input ** 2
        for layer in self.layers: 
            min_w, max_w = - 1 / np.sqrt(input_size + layer.size),  1 / np.sqrt(input_size + layer.size)
            layer.init_weights(input_size, (-min_w, max_w))
            input_size = layer.size

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

    def fit(self, 
        X_train, y_train, 
        X_val = [], y_val = [],
        X_test = [], y_test =[], 
        epoch: int = 100, size_minibatch: float = .8):
        
        X_train = X_train.reshape(-1, self.input ** 2)

        losses = []
        loss_val = []
        loss_test = []

        val = False
        test = False

        if not ((len(X_val) == 0) or (len(y_val) == 0)):
            X_val = X_val.reshape(-1, self.input ** 2)
            val = True

        if not ((len(X_test) == 0) or (len(y_test) == 0)):
            X_test = X_test.reshape(-1, self.input ** 2)
            test = True

        if "W" not in self.layers[0].__dict__.keys():
            self.init_weights()

        for ep in range(epoch):
            if val:
                ypred_val = self.forward_pass(X_val)
                loss_val_ep = loss[self.loss](y_val, ypred_val)
                loss_val.append(loss_val_ep)

            y_pred = self.forward_pass(X_train)
            loss_ep = loss[self.loss](y_train, y_pred)

            self.backward_pass(y_train, y_pred)
            self.update_weights()

            print_epoch_scores(ep+1, loss_ep, loss_val_ep) if val else print_epoch_scores(ep+1, loss_ep)
            losses.append(loss_ep)


        if test:
            ypred_test = self.forward_pass(X_test)
            loss_test.append(loss['cross_entropy'](y_test, ypred_test))
        

        plt.plot(range(1, epoch + 1), losses, c='b', label='train loss')
        if val: plt.plot(range(1, epoch + 1), loss_val, c='r', label='val loss')
        plt.legend()
        plt.show()

    def predict(self, X):
        X = X.reshape(-1, self.input ** 2)
        return self.forward_pass(X)

    def print_config(self):
        width = 40
        print_empty_tab_line(width)
        print_tab_line('GLOBAL', width=width)
        print_empty_tab_line(width)
        for k, v in self.global_config.__dict__.items():
            if not k in ['input', 'layers', 'type']:
                print_tab_line(k, str(v), width)
        print_empty_tab_line(width)
        print_tab_line('LAYERS', width=width)
        print_empty_tab_line(width)

        print_tab_line('input', str(self.layers_config.input), width=width)
        k = 1
        for layer in self.layers_config.hidden_layers:
            print_tab_line(f"layer {k}", width=width)
            k += 1
            for key, v in layer.items():
                print_tab_line('', f"{key}: {str(v)}", width=width)
        print_tab_line('type', str(self.layers_config.type), width=width)
        print_empty_tab_line(width)