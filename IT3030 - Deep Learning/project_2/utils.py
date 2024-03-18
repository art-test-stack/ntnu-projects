from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

def fit(
        model: object, 
        train_set: tuple[torch.Tensor], 
        val_set: tuple[torch.Tensor], 
        opt: object, 
        loss_func, 
        num_epochs: int = 10,
        device = torch.device("mps"),
        
    ) -> tuple[object, list, list, object]:
    X_train, y_train = train_set
    X_val, y_val = val_set
    losses = []
    val_loss = []
    for epoch in tqdm(range(num_epochs)):
        print("val loss")
        val_loss.append(loss_func(model(X_val.to(device)), y_val.to(device)).to("cpu").detach().numpy())

        mean_loss_batch = 0
        for k in range(len(X_train)):
            print("output")
            outputs = model(X_train[k,:,:].to(device))

            print("lossep")
            loss_ep = loss_func(outputs.to(device), y_train[k,:,:].to(device))
            mean_loss_batch += loss_ep.to("cpu").item()
            print("grad")
            opt.zero_grad()
            loss_ep.backward()
            opt.step()

        losses.append(mean_loss_batch/len(X_train))

        print(f"Epoch: {epoch+1}, loss: {losses[epoch]}, val_loss: {val_loss[epoch]}")

    return model, losses, val_loss, opt

# def fit_with_replace_previous_y(
#         model: object, 
#         train_set: tuple[torch.Tensor], 
#         val_set: tuple[torch.Tensor], 
#         opt: object, 
#         loss_func, 
#         num_epochs: int = 10
#     ) -> tuple[object, list, list, object]:
#     X_train, y_train = train_set
#     X_val, y_val = val_set
#     losses = []
#     val_loss = []
#     for epoch in tqdm(range(num_epochs)):
#         print("val loss")
#         val_loss.append(loss_func(model(X_val), y_val).detach().numpy())

#         print("output")
#         outputs = model(X_train)

#         print("lossep")
#         loss_ep = loss_func(outputs, y_train)
#         losses.append(loss_ep.detach().numpy())

#         print("grad")
#         opt.zero_grad()
#         loss_ep.backward()
#         opt.step()
#         print(f"Epoch: {epoch+1}, loss: {losses[epoch]}, val_loss: {val_loss[epoch]}")
#     return model, losses, val_loss, opt


def predict(model: object, outputScaler: object, test_set: tuple[torch.Tensor], time_delta_shifting: int = 0):
    X_test, y_test = test_set

    model.eval()
    y_pred = model(X_test).detach().numpy()
    
    if len(y_pred.shape) == 2:
        return outputScaler.inverse_transform(y_test).reshape(-1), outputScaler.inverse_transform(y_pred).reshape(-1)
    
    else:
        return outputScaler.inverse_transform(y_test[:,-1,:]).reshape(-1), outputScaler.inverse_transform(y_pred[:,-1,:]).reshape(-1)


def forecast(model: object, outputScaler: object, test_set: tuple[torch.Tensor], time_delta_shifting: int = 0, duration: int = 24):
    index_start_forecast = np.random.randint(0, len(X_test))
    X_test, y_test = test_set


    model.eval()
    
    y_pred = model(X_test).detach().numpy()
    
    if len(y_pred.shape) == 2:
        return outputScaler.inverse_transform(y_test).reshape(-1), outputScaler.inverse_transform(y_pred).reshape(-1)
    
    else:
        return outputScaler.inverse_transform(y_test[:,-1,:]).reshape(-1), outputScaler.inverse_transform(y_pred[:,-1,:]).reshape(-1)


def make_forecast(
        y_pred: torch.Tensor, 
        df_target: pd.DataFrame, 
        seq_len: int = 0, 
        forecast_duration: int = 24, 
        time_delta_shifting: int = 0
    ) -> None:
    index_start_forecast = np.random.randint(0, len(y_pred))
    dt = np.max([time_delta_shifting, seq_len])
    forecast_array = y_pred[index_start_forecast:index_start_forecast+ forecast_duration]
    x_axis_forecast = df_target[index_start_forecast + dt:index_start_forecast + dt + forecast_duration]['timestamp']

    df_forecast = pd.DataFrame({
        'timestamp': x_axis_forecast.values, 
        'consumption_forecast': forecast_array
        })
    df_forecast.index = x_axis_forecast.index

    df_result = df_target[index_start_forecast:index_start_forecast + dt + forecast_duration]
    df_result['forecast'] = df_forecast['consumption_forecast']

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