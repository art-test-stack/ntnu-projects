import pandas as pd
import numpy as np

def add_month_columns(
        df_input: pd.DataFrame, 
        params: dict = {}, 
        l: list[str] = []
    ) -> tuple[pd.DataFrame, list]:
    """
    Add a boolean column for each months if
    the measure have been done during this month
    """
    df = df_input.copy()
    dates_data = pd.concat([df, pd.get_dummies(df['timestamp'].dt.month, prefix='month')], axis=1)
    month_names = {1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june',
                7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'}
    df_out = dates_data.rename(columns={f'month_{month_num}': month_names[month_num] for month_num in range(1, 13)})
    return df_out, l 

def get_season(date):
    """
    Return the season of the date as input
    """
    if date.month in [3, 4, 5]:
        return 'spring'
    elif date.month in [6, 7, 8]:
        return 'summer'
    elif date.month in [9, 10, 11]:
        return 'fall'
    else:
        return 'winter'
    
def add_season_columns(
        df_input: pd.DataFrame, 
        params: dict = {}, 
        l: list[str] = []
    ) -> tuple[pd.DataFrame, list]:
    """
    Add a boolean column for each season if the 
    measure have been done during this season
    """
    df = df_input.copy()
    df['season'] = df['timestamp'].apply(get_season)
    df = pd.concat([df, pd.get_dummies(df['season'], prefix='season')], axis=1)
    return df.drop(columns=['season']), l
    
def add_hour_columns(
        df_input: pd.DataFrame, 
        params: dict = {}, 
        l: list[str] = []
    ) -> tuple[pd.DataFrame, list]:
    """
    Add a boolean column for each hour if the measure
    have been done during this hour
    """
    df = df_input.copy()
    return pd.concat([df, pd.get_dummies(df['timestamp'].dt.hour, prefix='hour')], axis=1), l

def change_timestamp_to_sin(
        df: pd.DataFrame, 
        params: dict = {}, 
        l: list[str] = []
    ) -> tuple[pd.DataFrame, list]:
    """
    Add a float column for each timestamp which correspond 
    to the value of sin(date in input) with a yearly period
    """
    df = df.copy()
    df['timestamp_sin'] = np.sin((pd.to_datetime(df['timestamp']).astype(int) // (3600 * 10 ** 9) - 414888) / ( 365 * 24 ))
    return df, l

def pick_location_data(
        df: pd.DataFrame, 
        params: dict = {}, 
        l: list[str] = []
    ) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    loc = params['loc']

    if type(loc) == int:
        loc = [loc]
    
    columns_to_drop = []
    for k in range(1, 6):
        if k not in loc: 
            columns_to_drop.append(f"NO{k}_consumption")
            columns_to_drop.append(f"NO{k}_temperature")
        
    return df.drop(columns=columns_to_drop), l

def shift_data(
        df: pd.DataFrame, 
        params: dict = {}, 
        l: list = []
    ) -> tuple[pd.DataFrame, list]:
    """
    params: { 
        'shift_min': int, 
        'shift_max': int, 
        'column_to_shift': str, 
        'new_column_name': str | None
        }
    """
    df = df.copy()
    shift_min = params['shift_min'] if 'shift_min' in params.keys() else 1
    shift_max = params['shift_max']
    column_to_shift = params['column_to_shift']
    new_column_name = params['new_column_name'] if 'new_column_name' in params.keys() else column_to_shift
    
    for k in range(shift_min, shift_max + 1):
        df[f"{new_column_name}_{k}_previous"] = df[column_to_shift].shift(k)
        l.append(f"{new_column_name}_{k}_previous")
    return df, l


def get_yesterday_target_mean(
        df: pd.DataFrame, 
        params: dict = {'target': 'NO1_consumption '}, 
        l: list = []
    ) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    
    target = params['target']

    df_temporary, columns_ = shift_data(df.copy(), params={'shift_min': 24, 'shift_max': 48, 'column_to_shift': target, 'new_column_name': 'target'})

    df[f"{target}_yesterday_mean"] = df_temporary[columns_].mean(axis=1)
    l.append(f"{target}_yesterday_mean")
    return df, l


def previous_y(
        df: pd.DataFrame, 
        params: dict = {'target': 'NO1_consumption'}, 
        l: list = []
    ) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    target = params['target']

    params = {
        'column_to_shift': target,
        'shift_max': 1
    }
    df, _ = shift_data(df, params, l.copy())

    columns = list(df.columns)

    columns.pop()
    column = f"{target}_1_previous"

    l.insert(0, column)
    columns.insert(0, column)
    df = df[columns]
    return df, l