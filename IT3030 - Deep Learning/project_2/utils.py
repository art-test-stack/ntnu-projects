from typing import Tuple, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch


def make_batch(dataset: Tuple[torch.Tensor, torch.Tensor], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = dataset
    start_index = np.random.randint(0, len(x))
    end_index = start_index + int(len(x) * batch_size / 100)

    batch_x = x[start_index:end_index, :, :] if end_index < len(x) else torch.cat((x[start_index:, :, :], x[:end_index % len(x), :, :]), 0)
    batch_y = y[start_index:end_index] if end_index < len(x) else torch.cat((y[start_index:], y[:end_index % len(x)]), 0)
    
    return batch_x, batch_y

def fit(
        model: object, 
        train_set: Tuple[torch.Tensor], 
        val_set: Tuple[torch.Tensor], 
        opt: object, 
        loss_func, 
        num_epochs: int = 10,
        device = torch.device("mps"),
        batch_size: int = 100,
        plot_epoch_losses: bool = False
    ) -> Tuple[object, list, list, object]:
    X_train, y_train = train_set
    X_val, y_val = val_set
    losses = []
    val_loss = []

    print('device:', device)
    
    for epoch in tqdm(range(num_epochs)):
        val_loss.append(loss_func(model(X_val.to(device)), y_val.to(device)).to("cpu").detach().numpy())

        x, y = make_batch((X_train, y_train), batch_size=batch_size) if (batch_size > 0 and batch_size < 100) else (X_train, y_train)

        outputs = model(x.to(device))

        loss_ep = loss_func(outputs.to(device), y.to(device))

        losses.append(loss_ep.to("cpu").detach().numpy())
        opt.zero_grad()
        loss_ep.backward()
        opt.step()

        print(f"Epoch: {epoch+1}, loss: {losses[epoch]}, val_loss: {val_loss[epoch]}") if plot_epoch_losses else None

    return model, losses, val_loss, opt


def predict(
        model: object, 
        outputScaler: object, 
        test_set: Tuple[torch.Tensor, torch.Tensor],
        device = torch.device("mps"),
        ):
    X_test, y_test = test_set

    model.eval()
    y_pred = model(X_test.to(device)).cpu().detach().numpy()
    
    if len(y_pred.shape) == 2:
        return outputScaler.inverse_transform(y_test).reshape(-1), outputScaler.inverse_transform(y_pred).reshape(-1)
    
    else:
        return outputScaler.inverse_transform(y_test[:,-1,:]).reshape(-1), outputScaler.inverse_transform(y_pred[:,-1,:]).reshape(-1)


def make_forecast(
        y_pred: torch.Tensor, 
        df_target: pd.DataFrame, 
        seq_len: int = 0, 
        forecast_duration: int = 24,
    ) -> None:
    dt = seq_len - 1
    index_start_forecast = np.random.randint(0, len(y_pred) - forecast_duration) if len(y_pred) - forecast_duration > 0 else 0

    forecast_array = y_pred[index_start_forecast:index_start_forecast + forecast_duration]
    x_axis_forecast = df_target[index_start_forecast + dt:index_start_forecast + dt + forecast_duration]['timestamp']

    df_forecast = pd.DataFrame({
        'timestamp': x_axis_forecast.values, 
        'consumption_forecast': forecast_array
        })
    df_forecast.index = x_axis_forecast.index

    df_result = df_target[index_start_forecast:index_start_forecast + dt + forecast_duration]
    df_result['forecast'] = df_forecast['consumption_forecast']

    print(len(df_result))
    df_result.set_index('timestamp').plot(figsize=(10,5))
    plt.legend()
    plt.show()


def plot_error_by_hour_for_test_set(y_test: torch.Tensor, y_pred: torch.Tensor, start_hour: int = 12) -> None:
    error_by_hours = [ [] for _ in range(24) ]
    current_hour = start_hour
    for k in range(len(y_test)):
        error_by_hours[current_hour].append(np.abs(y_test[k] - y_pred[k]))
        current_hour = (current_hour + 1) % 24

    average = [ np.mean(error) for error in error_by_hours]
    std_dev = [ np.std(error) for error in error_by_hours]
    x_axis = range(24)
    
    plt.figure(figsize=(12, 7))
    plt.errorbar(x_axis, average, yerr=std_dev, fmt='o')
    plt.xlabel('hours')
    plt.ylabel('abs error')
    plt.show()


def make_forecast_replace_previous_y(
        df_target: pd.DataFrame,
        model: object, 
        outputScaler: object, 
        test_set: Tuple[torch.Tensor, torch.Tensor],
        seq_len: int = 0,
        device = torch.device("mps"),
        forecast_duration: int = 24,
    ):
    df_target = df_target.copy()
    X_test, y_test = test_set
    
    dt = seq_len - 1
    index_start_forecast = np.random.randint(0, len(y_test) - forecast_duration)
    
    model.eval()
    
    y_pred = []
    t0 = index_start_forecast
    for t in range(t0, t0 + forecast_duration):
        x = X_test[t, :, :].view(1, seq_len, -1)
        y = model(x.to(device))
        for i in range(0, forecast_duration):
            X_test[t + i + 1, seq_len - forecast_duration + i, 0] = y[:, -1, :]
        y_pred.append(y[:, -1, :].cpu().detach().numpy())

    y_pred = np.array(y_pred).reshape(1, -1)

    y_pred = outputScaler.inverse_transform(y_pred).reshape(-1)
    
    make_forecast(
        y_pred,
        df_target[index_start_forecast:index_start_forecast + dt + forecast_duration].reset_index(drop=True),
        seq_len,
        forecast_duration
    )

from torchmetrics.regression import MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError

def RootMeanSquaredError(target, pred):
    return torch.sqrt(MeanSquaredError()(target, pred))

def summary_bar_plot(
        preds: Dict,
        targets: Dict,
        losses: Dict = {
            'MAPE': MeanAbsolutePercentageError(), 
            'MAE': MeanAbsoluteError(), 
            'RMSE': RootMeanSquaredError
            },
    ):
    assert preds.keys() == targets.keys()
    
    loss_values = { model: [
        1 * loss(targets[model], preds[model]).detach().numpy() if not (metric == 'MAPE') else 100 * loss(targets[model], preds[model]).detach().numpy() 
        for metric, loss in losses.items()
        ] for model in preds.keys() }

    bar_width = 0.1

    index = np.arange(len(losses.keys()))

    plt.figure(figsize=(12, 6))
    axis_x = index - bar_width

    for model in loss_values.keys():
        plt.bar(axis_x, loss_values[model], bar_width, label=model)
        axis_x += bar_width
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Grouped Bar Plot of Different Metrics')
    plt.xticks(index, losses.keys())
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()