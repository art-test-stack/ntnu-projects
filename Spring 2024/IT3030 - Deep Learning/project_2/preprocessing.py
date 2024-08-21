import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from sklearn.preprocessing import StandardScaler

from copy import copy
from features import *


# -------------------------------------------------------
#               ADJUST DATA FOR PREPROCESSING
# -------------------------------------------------------

def manage_anomalies(df: pd.DataFrame, threshold: float = 3.5, interpolation_method: str = 'linear'):
    df = df.copy()
    df_anomalies = df.copy()

    columns = df.drop(columns='timestamp', inplace=False).columns

    mean = df[columns].mean()
    mean['timestamp'] = df['timestamp']
    std = df[columns].std()
    std['timestamp'] = df['timestamp']

    for column in columns:
        th = threshold * std[column]
        anomalies = (df[column] - mean[column]).abs() < th

        df_anomalies[column] = df[column][anomalies]

    return df_anomalies.interpolate(method=interpolation_method)


def manage_minmax_values(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    columns = df_train.columns
    for column in columns:
        df_val[column][df_val[column] > df_train.max()[column]] = df_train.max()[column]
        df_val[column][df_val[column] < df_train.min()[column]] = df_train.min()[column]

        df_test[column][df_test[column] > df_train.max()[column]] = df_train.max()[column]
        df_test[column][df_test[column] < df_train.min()[column]] = df_train.min()[column]
    
    return df_train, df_val, df_test


def make_sequences(x, y, seq_len=9, dt = 1):
    "Make sequences for recurrent networks"
    num_samples = x.shape[0]

    num_sequences = num_samples - seq_len + 1

    sequences = []
    targets = []

    for i in range(num_sequences):
        seq = x[i:i+seq_len]
        target = y[i+dt:i+seq_len+dt]
        sequences.append(seq)
        targets.append(target)

    sequences_padded = pad_sequence(sequences, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True)

    sequences_tensor = torch.tensor(sequences_padded, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_padded, dtype=torch.float32)
    return sequences_tensor, targets_tensor

def split_dataset_by_proportions(
        df_input: pd.DataFrame, 
        df_output: pd.DataFrame, 
        sizes: tuple[int] = (80, 20, 10), 
        seq_len: int = 0,
        ):
    assert len(df_input) == len(df_output), "Sizes should be the same"
    assert sum(sizes) == 100, "Proportions should be equal to 100"

    train_end_index = int(sizes[0] * len(df_input) / 100)
    val_end_index = train_end_index + int(sizes[1] * len(df_input) / 100)

    X_train = df_input[:train_end_index]
    Y_train = df_output[:train_end_index]

    X_val = df_input[train_end_index - seq_len:val_end_index]
    Y_val = df_output[train_end_index - seq_len:val_end_index]

    X_test = df_input[val_end_index - seq_len:]
    Y_test = df_output[val_end_index - seq_len:]
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


# -------------------------------------------------------
#           GENERAL FUNCTION FOR PREPROCESSING
# -------------------------------------------------------


def general_preprocessing(
        df_raw: pd.DataFrame,
        target_column: str = 'NO1_consumption',
        features_to_add: list[tuple] = [(add_season_columns, {}) ],
        sets_sizes: tuple[int] = (70, 20, 10),
        scalerInputMethod: object = StandardScaler(),
        scalerOutputMethod: object = StandardScaler(),
        scale_output: bool = False,
        is_scaler_fitted: bool = False,
        seq_len: int = 48,
        features_to_scale: list = ['NO1_temperature'],
        dt_output_input: int = 0,
        anomalies_threshold: float = 3.5,
        interpolation_method: str = 'linear',
        do_manage_anomalies: bool = True,
        do_manage_minmax_values: bool = True
    ) -> tuple[
            tuple[torch.Tensor, torch.Tensor], 
            tuple[torch.Tensor, torch.Tensor], 
            tuple[torch.Tensor, torch.Tensor], 
            tuple[object, object]
    ]:
    features_to_scale = features_to_scale.copy()
    df = df_raw.copy()

    if do_manage_anomalies:
        df = manage_anomalies(df, threshold=anomalies_threshold, interpolation_method=interpolation_method)
   
    for func, params in features_to_add:
        df, features_to_scale = func(df, params, features_to_scale)

    df = df.dropna()
    df = df.replace(True, 1)
    df = df.replace(False, 0)

    # SPLIT TARGET AND INPUT
    df_X = df.drop(columns=[target_column, 'timestamp'], inplace=False)
    df_y = df[[target_column]]

    # SPLIT DATASET
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_dataset_by_proportions(
        df_X, df_y, sizes=sets_sizes, seq_len=seq_len)

    df_target = df[['timestamp', target_column]][len(df) - len(X_test):]

    print("FEATURES:\n", X_train.columns)
    
    # SCALE INPUT VALUES

    X_train[features_to_scale] = scalerInputMethod.fit_transform(X_train[features_to_scale].values) if not is_scaler_fitted else  scalerInputMethod.transform(X_train[features_to_scale].values) 
    X_val[features_to_scale] = scalerInputMethod.transform(X_val[features_to_scale])
    X_test[features_to_scale] = scalerInputMethod.transform(X_test[features_to_scale])

    # SCALE OUTPUT VALUES
    if scale_output:
        Y_train[[target_column]] = scalerOutputMethod.fit_transform(Y_train[[target_column]]) if not is_scaler_fitted else scalerOutputMethod.transform(Y_train[[target_column]])
        Y_test[[target_column]] = scalerOutputMethod.transform(Y_test[[target_column]])
        Y_val[[target_column]] = scalerOutputMethod.transform(Y_val[[target_column]])

    # MANAGE MIN-MAX VALUES
    if do_manage_minmax_values:
        X_train, X_val, X_test = manage_minmax_values(X_train, X_val, X_test)
        Y_train, Y_val, Y_test = manage_minmax_values(Y_train, Y_val, Y_test)

    # TRANSFORM DataFrames TO torch.Tensor
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)

    y_train = torch.tensor(Y_train.values, dtype=torch.float32)
    y_test = torch.tensor(Y_test.values, dtype=torch.float32)
    y_val = torch.tensor(Y_val.values, dtype=torch.float32)

    # MAKE SEQUENCES
    if seq_len > 0:
        X_train, y_train = make_sequences(X_train, y_train, seq_len=seq_len, dt=dt_output_input)
        X_test, y_test = make_sequences(X_test, y_test, seq_len=seq_len, dt=dt_output_input)
        X_val, y_val = make_sequences(X_val, y_val, seq_len=seq_len, dt=dt_output_input)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (scalerInputMethod, scalerOutputMethod), df_target
