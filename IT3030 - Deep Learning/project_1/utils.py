import yaml
from enum import Enum


# -------------------------------------------
#           NEURAL NETWORK PARAMETERS
# -------------------------------------------


class Loss(Enum):
    cross_entropy: str = 'cross_entropy'
    mse: str = 'mse'

class Regularizator(Enum):
    L1: str = 'L1'
    L2: str = 'L2'
    none = None

class ActivationFunction(Enum):
    sigmoid: str = 'sigmoid'
    tanh: str = 'tanh'
    relu: str = 'relu'
    linear: str = 'linear'
    
class OutputActFunction(Enum):
    softmax: str = 'softmax'

class HiddenLayerConfig:
    size: int = 10
    act: str = 'relu'
    wr: tuple # | str
    lrate: float = 0.01

class InputLayerConfig:
    input: int = 20

class OutputLayerConfig:
    type: OutputActFunction = 'softmax'


class GlobalConfig:

    def __init__(self, **params):
        self.loss: Loss = Loss.cross_entropy.value
        self.lrate: float = 0.1
        self.wreg: float = 0
        self.wrt: Regularizator = None
        for k, v in params.items():
            setattr(self, k, v)

class LayersConfig:
    input: InputLayerConfig.input
    layers: list[HiddenLayerConfig]
    type: OutputLayerConfig.type

    def __init__(self, **params):
        for k, v in params.items():
            setattr(self, k, v)

class DatasetConfig: 
    def __init__(self, **params):
        self.load = True
        self.name = 'dataset_2024-02-13_750_9_50'

        for k, v in params.items():
            setattr(self, k, v)


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

basic_config = read_config('configs/basic_config.yaml')

def open_config(file_path = 'configs/basic_config.yaml'):
    config = read_config(file_path)

    global_config = GlobalConfig(**config['GLOBAL']) if 'GLOBAL' in config.keys() and config['GLOBAL'] is not None else GlobalConfig(**basic_config['GLOBAL'])
    layers_config = LayersConfig(**config['LAYERS']) if 'LAYERS' in config.keys() and config['LAYERS'] is not None else GlobalConfig(**basic_config['LAYERS'])
    dataset_config = DatasetConfig(**config['DATASET']) if 'DATASET' in config.keys() and config['DATASET'] is not None else GlobalConfig(**basic_config['DATASET'])
    return global_config, layers_config, dataset_config


# -------------------------------------------
#           DATASET UTILS FUNCTIONS
# -------------------------------------------

import doodler_forall as dd
import numpy as np
from pathlib import Path
from datetime import datetime

_doodle_image_types = dd._doodle_image_types_


def generate_dataset_folder_name(size, nb_classes, n):
    address = Path('datasets')
    name_folder = f"dataset_{str(datetime.now().date())}_{size}_{nb_classes}_{n}"

    path = address / name_folder
    if not path.exists():
        path.mkdir()
        return name_folder
    
    k = 1
    folder = name_folder + f"_v{k}"
    path = address / folder
    while path.exists():
        k += 1
        folder = name_folder + f"_v{k}"
        path = address / folder
    
    path.mkdir()

    return folder

def split_dataset(dataset, proportions = (70, 20, 10)):
    assert sum(proportions) == 100, "Proportions do not make 100%"

    train_size = int(proportions[0] * len(dataset[0]) / 100)
    test_size = int(proportions[1] * len(dataset[0]) / 100)

    inputs, targets, labels, _, _ = dataset
    # dataset = (inputs.astype(np.float32), targets, target_labels,(rows,cols),flat)
    X_train = inputs[:train_size]
    Y_train = targets[:train_size]
    labels_train = labels[:train_size]

    X_test = inputs[train_size:train_size + test_size]
    Y_test = targets[train_size:train_size + test_size]
    labels_test = labels[train_size:train_size + test_size]

    X_val = inputs[train_size + test_size:]
    Y_val = targets[train_size + test_size:]
    labels_val = labels[train_size + test_size:]
    
    return (X_train, Y_train, labels_train), (X_test, Y_test, labels_test), (X_val, Y_val, labels_val)

def save_data_in_folder(address, data_subset, X, Y, labels):
    
    np.save(address / f"X_{data_subset}.npy", X)
    np.save(address / f"Y_{data_subset}.npy", Y)
    np.save(address / f"labels_{data_subset}.npy", labels)

def create_dataset(proportions = (70, 20, 10), 
                   dataset_size = 100, image_size = 30, noise = .1, wr=[0.2,0.4],hr=[0.2,0.4],
                   flattening = False, centered = False, image_types = _doodle_image_types, show_images = False):
    dataset_name = generate_dataset_folder_name(dataset_size, len(image_types), image_size)
    
    dataset = dd.gen_doodle_cases(count=dataset_size,rows=image_size,cols=image_size,imt=image_types, 
                hr=hr,wr=wr, nbias=noise,cent=centered, show=False,
                one_hots=True,auto=False, flat=flattening,bkg=0, d4=False, fog=1, fillfrac=None, fc=(1,1),
                gap=1,multi=False,mono=True, dbias=0.7,poly=(4,5,6))
    
    (X_train, Y_train, labels_train), (X_test, Y_test, labels_test), (X_val, Y_val, labels_val) = split_dataset(dataset, proportions)

    save_data_in_folder(Path('datasets') / dataset_name, 'train', X_train, Y_train, labels_train)
    save_data_in_folder(Path('datasets') / dataset_name, 'test', X_test, Y_test, labels_test)
    save_data_in_folder(Path('datasets') / dataset_name, 'val', X_val, Y_val, labels_val)
    
    if show_images: show_img_cases(dataset_name)
    return dataset_name

def load_subsets(address, data_subset):
    X = np.load(address / f"X_{data_subset}.npy")
    Y = np.load(address / f"Y_{data_subset}.npy")
    labels = np.load(address / f"labels_{data_subset}.npy")

    return (X, Y, labels)

def load_dataset_from_folder_name(dataset_name): 

    address = Path('datasets') / dataset_name

    training_set = load_subsets(address, 'train')
    testing_set = load_subsets(address, 'test')
    val_set = load_subsets(address, 'val')

    return training_set, testing_set, val_set

def load_dataset(dataset_config):
    dataset_name = dataset_config.name if dataset_config.load else create_dataset(**dataset_config.params)
    training_set, testing_set, val_set = load_dataset_from_folder_name(dataset_name)
    return training_set, testing_set, val_set

def show_img_cases(dataset_name, nb_cases = 10):
    sets = load_dataset_from_folder_name(dataset_name)
    nb_cases = np.minimum(nb_cases, len(sets[0][0]))
    
    images = []
    labels = []

    index = []
    for _ in range(np.minimum(nb_cases, 10)):
        set = sets[np.random.randint(0, 2)]
        rd_img_index = np.random.randint(0, set[0].shape[0] - 1)
        while rd_img_index in index:
            rd_img_index = np.random.randint(0, set[0].shape[0] - 1)
        index.append(rd_img_index)
        images.append(set[0][rd_img_index])
        labels.append(set[2][rd_img_index])

    cases = (images, '', labels, '', '')
    dd.show_doodle_cases(cases)

