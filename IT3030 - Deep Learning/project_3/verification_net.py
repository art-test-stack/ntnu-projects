
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from stacked_mnist import StackedMNIST

from typing import Tuple

class VerificationNet(nn.Module):
    def __init__(
        self, 
        force_learn: bool = False, 
        file_name: str = "./models/verification_model_torch",
        device = torch.device("mps"),
    ) -> None:
        """
        Define model and set some parameters.
        The model is  made for classifying one channel only -- if we are looking at a
        more-channel image we will simply do the thing one-channel-at-the-time.
        """
        super().__init__()
        
        self.force_relearn = force_learn
        self.file_name = file_name
        self.device = device

        layers = [
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3)), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25)
        ]

        for _ in range(2):
            layers.append(nn.Conv2d(64, 64, kernel_size=(3, 3)))
            layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
            layers.append(nn.Dropout(0.25))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(128, 10))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

        self.loss = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.parameters(), lr=1e-4)

        self.done_training = self.load_weights()
        self.to(device)

        self.losses = []
        self.val_loss = []

    def load_weights(self):
        # noinspection PyBroadException
        try:
            self.model.load_state_dict(torch.load(self.file_name))
            # print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(
                f"Could not read weights for verification_net from file. Must retrain..."
            )
            done_training = False

        self.done_training = done_training

    def train(self, trainset: StackedMNIST, valset: StackedMNIST, epochs: int = 10, batch_size: int = 256) -> bool:
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
        self.load_weights()

        if self.force_relearn or self.done_training is False:
            train_set = DataLoader(trainset, shuffle=True, batch_size=batch_size)
            val_set = DataLoader(valset, shuffle=True, batch_size=batch_size)
            self.fit(
                train_set=train_set,
                validation_set=val_set,
                epochs=epochs,
            )
            torch.save(self.model.state_dict(), self.file_name)
            self.done_training = True

        return self.done_training
    
    def forward(self, x):
        out = self.model(x)
        return out
    
    def fit(
            self,
            train_set: DataLoader, 
            validation_set: DataLoader, 
            epochs: int = 10,
        ) -> Tuple[object, list, list, object]:
        
        device = self.device

        for _ in tqdm(range(epochs)):
            val_loss_ep = []
            for _, batch in enumerate(validation_set, 0):
                x_val, y_val, _ = batch
                x_val = x_val[:,[0],:,:]
                y_val = y_val.reshape(y_val.shape[0],-1,10)[:, 0, :]
                outval = self(x_val.to(device))
                val_loss_ep.append(self.loss(outval, y_val.to(device)).to("cpu").item())
            
            self.val_loss.append(sum(val_loss_ep)/len(val_loss_ep))

            loss_ep = []
            for _, batch in enumerate(train_set, 0):
                x, y, _ = batch
                x = x[:,[0],:,:]
                y = y.reshape(y.shape[0],-1,10)[:, 0, :]

                self.opt.zero_grad()

                output = self(x.to(device))
                loss_batch = self.loss(y.to(device), output.to(device))

                loss_batch.backward()
                self.opt.step()
                
                loss_ep.append(loss_batch.to("cpu").item())
            self.losses.append(np.mean(loss_ep))


    def predict(self, data: torch.Tensor) -> tuple:
        data = data.to(self.device)
        num_channels = data.shape[1]

        if self.done_training is False:
            # Model is not trained yet...
            raise ValueError("Model is not trained, so makes no sense to try to use it")

        predictions = np.zeros((data.shape[0],))
        beliefs = np.ones((data.shape[0],))
        for channel in range(num_channels):
            channel_prediction = self.model(data[:, [channel], :, :]).to('cpu').detach().numpy()
            beliefs = np.multiply(beliefs, np.max(channel_prediction, axis=1))
            predictions += np.argmax(channel_prediction, axis=1) * np.power(10, channel)
        return predictions, beliefs

    def check_class_coverage(
        self, data: torch.Tensor, tolerance: float = 0.8
    ) -> float:
        """
        Out of the total number of classes that can be generated, how many are in the data-set?
        I'll only count samples for which the network asserts there is at least tolerance probability
        for a given class.
        """
        num_classes_available = np.power(10, data.shape[1])
        predictions, beliefs = self.predict(data)

        predictions = predictions[beliefs >= tolerance]

        coverage = float(len(np.unique(predictions))) / num_classes_available
        return coverage

    def check_predictability(
        self, data: torch.Tensor, correct_labels: np.array = None, tolerance: float = 0.8
    ) -> tuple:
        """
        Out of the number of data points retrieved, how many are we able to make predictions about?
        ... and do we guess right??

        Inputs here are
        - data samples -- size (N, 28, 28, color-channels)
        - correct labels -- if we have them. List of N integers
        - tolerance: Minimum level of "confidence" for us to make a guess

        """
        predictions, beliefs = self.predict(data=data)
        predictions = predictions[beliefs >= tolerance]
        predictability = len(predictions) / len(data)

        if correct_labels is not None:
            # Drop those that were below threshold
            correct_labels = correct_labels[beliefs >= tolerance]
            accuracy = np.sum(predictions == correct_labels) / len(data)
        else:
            accuracy = None

        return predictability, accuracy
