import numpy as np
from utils import GlobalConfig, LayersConfig, read_config, adapt_config

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def print_empty_tab_line(width=40):
    print(f"|{'-'* width}", end='|\n')

# def print_tab_line(content, width = 40):
#     print(f"| {content}{' '*(width - len(content) - 1)}", end='|\n')

def print_tab_line(row1, row2 = '', width = 40):
    print(f"| {row1}{' '*(9 - len(row1) - 1)} | {row2}{' '*(width - len(row2) - 12)}", end='|\n')


class MLP:
    def __init__(
            self, 
            global_config: GlobalConfig,
            layers_config: LayersConfig,
            show_config: bool = True
        ):
        for k, v in global_config.__dict__.items():
            setattr(self, k, v)
        
        self.init_layers(layers_config)
        if show_config: 
            self.global_config = global_config
            self.layers_config = layers_config
            self.print_config()

    def init_weights(self, dim):
        W = np.array([])
        for layer in self.layers:
            if layer.wr == 'glorot':
                pass
            
            else: 
                pass

    def init_layers(self, layers_config):
        self.input = layers_config.input
        self.type = layers_config.type
        self.layers = layers_config.hidden_layers

    

            
    def split_minibatch(self):
        pass

    def forward_pass(self, X_train, y_train, epoch, size_minibatch: float = .8):
        pass
    
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