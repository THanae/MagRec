import pandas as pd
import numpy as np
from datetime import timedelta


def get_moving_average(data_column: pd.Series, minutes: int = 10) -> pd.Series:
    moving_average = pd.Series(np.zeros_like(data_column.values), index=data_column.index)
    for index, value in data_column.iteritems():
        start_time = index - timedelta(minutes=minutes)
        end_time = index + timedelta(minutes=minutes)
        moving_average.loc[index] = np.mean(data_column.loc[start_time:end_time])
    return moving_average


def get_derivative(data_column: pd.Series) -> pd.Series:
    return data_column.diff() / data_column.index.to_series().diff().dt.total_seconds()


def get_outliers(data_column: pd.Series, minutes: float = 10, standard_deviations: float = 2,
                 ignore_minutes_around: float = 0, reference='median') -> pd.Series:
    outliers = pd.Series(np.zeros_like(data_column.values), index=data_column.index)

    for index, value in data_column.iteritems():
        if not ignore_minutes_around:
            start_time = index - timedelta(minutes=minutes)
            end_time = index + timedelta(minutes=minutes)
            values_to_consider = data_column[data_column.index.values != index].loc[start_time:end_time]
        else:
            left_interval_start = index - timedelta(minutes=(minutes + ignore_minutes_around))
            left_interval_end = index - timedelta(minutes=ignore_minutes_around)
            right_interval_start = index + timedelta(minutes=ignore_minutes_around)
            right_interval_end = index + timedelta(minutes=(minutes + ignore_minutes_around))

            values_to_consider = pd.concat((data_column.loc[left_interval_start: left_interval_end],
                                            data_column.loc[right_interval_start: right_interval_end]))
            # print(values_to_consider)
        if reference == 'median':
            reference = values_to_consider.median()
        elif reference == 0:
            reference = 0
        if abs(value - reference) > standard_deviations * values_to_consider.std():
            outliers.loc[index] = value
            # outliers.loc[index] = abs(value - values_to_consider.median()) / values_to_consider.std()
        else:
            outliers.loc[index] = np.nan
    return outliers
