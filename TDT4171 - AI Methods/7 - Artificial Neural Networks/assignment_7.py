import numpy as np
import matplotlib.pyplot as plt

def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test

def loss_function(target, pred):
    return 1 / 2 * np.sum((target - pred) ** 2) / len(target)

def d_loss_function(target, pred):
    return (target - pred)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def tanh(x):
    return ( np.exp(-x) - np.exp(x) ) / ( np.exp(-x) + np.exp(x) ) 

def d_tanh(x):
    return 1 - x ** 2

class FeedForwardNetwork:
    def __init__(self, input_size = 2, hidden_layer_size: int = 2, wr = (-.1, .1)):
        self.size = hidden_layer_size
        self.w1 = np.random.uniform(wr[0], wr[1], size=(input_size, self.size))
        self.b1 = np.random.uniform(wr[0], wr[1], size=(1, self.size))
        self.w2 = np.random.uniform(wr[0], wr[1], size=(self.size))
        self.b2 = np.random.uniform(wr[0], wr[1], size=(1))

    def forward_pass(self, X, is_training = False):
        y1 = sigmoid(np.dot(X, self.w1) + self.b1)
        if is_training: self.y1 = y1
        return tanh(np.dot(y1, self.w2) + self.b2)
    
    def backward_pass(self, X_train, target, output):
        d_loss = d_loss_function(target, output)

        grad2 = d_loss * d_tanh(output)
        self.dw2 = (self.y1.T).dot(grad2) / len(self.y1)
        self.db2 = np.mean(grad2, axis=0)

        grad1 = grad2.reshape(-1, 1).dot(self.w2.reshape(-1, 1).T) * d_sigmoid(self.y1)
        self.dw1 = (X_train.T).dot(grad1) / len(X_train)
        self.db1 = np.mean(grad1, axis=0)

    def update_weights(self, learning_rate):
        self.w1 -= learning_rate * self.dw1
        self.b1 -= learning_rate * self.db1
        self.w2 -= learning_rate * self.dw2
        self.b2 -= learning_rate * self.db2

    def fit(self, X_train, y_train, X_val = [], y_val = [], learning_rate=0.1, epochs=100):
        losses = []
        val_losses = []

        has_val = False
        if len(X_val) > 0:
            has_val = True

        for _ in range(epochs):
            y = self.forward_pass(X_train, is_training=True)
            losses.append(loss_function(y_train, y))
            if has_val: val_losses.append(loss_function(y_val, self.forward_pass(X_val)))
            self.backward_pass(X_train, y_train, y)
            self.update_weights(learning_rate)

        return losses, val_losses

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    wr = (- np.sqrt( 6 / (2 + 2)), np.sqrt( 6 / (2 + 2))) # Xavier Glorot initialization

    # ffn = FeedForwardNetwork(hidden_layer_size=2, wr=wr)

    # train_losses, val_losses = ffn.fit(
    #     X_train=X_train, y_train=y_train, 
    #     X_val=X_test, y_val=y_test, 
    #     learning_rate=.5, epochs=10000
    #     )
    # test_loss = loss_function(target=y_test, pred=ffn.forward_pass(X_test))

    # learning_rates = [x / 100 if x <= 10 else x / 10 for x in range(1, 101)]

    learning_rates = [.001, .1, .5, .6, .7, .8, .9, 1.0, 1.5, 1.6, 1.8, 1.9, 2.0, 2.5, 10]
    test_losses = []
    train_val_losses = []
    ffns = []
    k = 0
    from tqdm import tqdm
    for lr in tqdm(learning_rates):
        ffns.append(FeedForwardNetwork(hidden_layer_size=2, wr=wr))
        train_val_losses.append(ffns[k].fit(
            X_train=X_train, y_train=y_train, 
            X_val=X_test, y_val=y_test, 
            learning_rate=lr, epochs=10000
            ))
        test_losses.append(loss_function(target=y_test, pred=ffns[k].forward_pass(X_test)))
        k += 1

    print('best learning rate:', learning_rates[np.argmin(test_losses)])

    ffn = ffns[np.argmin(test_losses)]
    train_losses, val_losses = train_val_losses[np.argmin(test_losses)]

    test_loss = loss_function(target=y_test, pred=ffn.forward_pass(X_test))

    print("Test loss:", test_loss)

    print('mean train diff', np.mean(np.abs(y_train - ffn.forward_pass(X_train))))
    print('mean test diff', np.mean(np.abs(y_test - ffn.forward_pass(X_test))))

    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.legend()
    plt.show()

    plt.plot(y_test, label='target')
    plt.plot(ffn.forward_pass(X_test), label='pred')
    plt.legend()
    plt.show()

    plt.plot(func(X_test), label='original')
    plt.plot(ffn.forward_pass(X_test), label='pred')
    plt.legend()
    plt.show()

    plt.plot(y_test - ffn.forward_pass(X_test), label='diff')
    plt.legend()
    plt.show()


    