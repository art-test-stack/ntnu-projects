import numpy as np


# ---------------------ACTIVATION FUNCTIONS---------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

act_function = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'linear': linear,
    'relu': relu 
}

# -------------DERIVATIVES OF ACTIVATION FUNCTIONS--------------

def d_sigmoid(x):
    return x * ( 1 - x )

def d_tanh(x):
    return 1 - x**2

def d_linear(x):
    return np.ones(x.shape)

def d_relu(x):
    return np.where(x<=0, 0, 1)

d_act_function = {
    'sigmoid': d_sigmoid,
    'tanh': d_tanh,
    'linear': d_linear,
    'relu': d_relu
}

# ------------------OUPUT ACTIVATION FUNCTION------------------

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def d_softmax(x):
    diag = []
    product = []
    for mini_x in x:
        diag.append(np.diag(mini_x))
        mini_x = mini_x.reshape(-1, 1)
        product.append(mini_x.dot(mini_x.T))
    diag = np.array(diag)
    product = np.array(product)
    return diag - product

# def d_softmax(x):
#     return x * ( 1 - x  )

type_function = {
    'softmax': softmax
}

d_type_function = {
    d_softmax
}

# -----------------------------LOSS-----------------------------

def cross_entropy(y, y_pred):
    eps = 1e-15
    loss = - 1 / y.shape[0] * np.sum(y * np.log( y_pred + eps ))
    return loss

def d_cross_entropy(y, y_pred):
    eps = 1e-15
    return np.sum(y) / np.sum(y_pred) - 1 * (y / (y_pred + eps))

def d_ce_s(y, y_pred):
    return - (y - np.sum(y, axis=1, keepdims=True) * y_pred)

def mse(y, y_pred):
    return 1 / y.shape[0] * np.sum((y - y_pred)**2)

def d_mse(y, y_pred):
    return (y - y_pred)

loss = {
    'cross_entropy': cross_entropy,
    'mse': mse
}

d_loss = {
    'cross_entropy': d_cross_entropy,
    'mse': d_mse
}
