import pandas as pd
import numpy as np
from datetime import timedelta


COORDINATES = ['x', 'y', 'z']


def get_moving_average(data_column: pd.Series, minutes: int = 10) -> pd.Series:
    """
    Get the average of the data around a certain point
    :param data_column: column to average
    :param minutes: minutes to the right and to the left that will be considered when taking the moving average
    :return:
    """
    moving_average = pd.Series(np.zeros_like(data_column.values), index=data_column.index)
    for index, value in data_column.iteritems():
        start_time = index - timedelta(minutes=minutes)
        end_time = index + timedelta(minutes=minutes)
        moving_average.loc[index] = np.mean(data_column.loc[start_time:end_time])
    return moving_average


def get_derivative(data_column: pd.Series) -> pd.Series:
    """
    Returns the derivative of the column.
    Derivative is given as difference of two consecutive data points divided by the time between them
    :param data_column: column to derive
    :return:
    """
    return data_column.diff() / data_column.index.to_series().diff().dt.total_seconds()


def get_outliers(data_column: pd.Series, minutes: float = 10, standard_deviations: float = 2,
                 ignore_minutes_around: float = 0, reference='median') -> pd.Series:
    """
    Find outliers in a given column
    :param data_column: column to analyse
    :param minutes: minutes during which the data will be considered for the outliers tests
    :param standard_deviations: standard deviations that will be used when comparing data points to surrounding points
    :param ignore_minutes_around: number of minutes around potential events (to the right and to the left) to ignore
    :param reference: reference to use in comparison (median of value to consider or 0)
    :return:
    """
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


def find_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """
    Find the columns of correlations.
    Consists in finding C, given by delta_b / std_b * delta_v / std_v, where the std are there for normalisation
    :param data: data to use
    :return:
    """
    coordinate_correlation_column_names = []

    for coordinate in COORDINATES:
        field_column_name = 'B' + coordinate
        v_column_name = 'vp_' + coordinate
        field_column = data[field_column_name].interpolate('time')
        v_column = data[v_column_name].interpolate('time')

        delta_b = get_derivative(field_column)
        delta_v = get_derivative(v_column)

        std_b = (data[field_column_name] - get_moving_average(data[field_column_name])).std()
        std_v = (data[v_column_name] - get_moving_average(data[v_column_name])).std()
        correlations = delta_b / std_b * delta_v / std_v

        column_name = 'correlation_{}'.format(coordinate)
        data[column_name] = correlations.abs().apply(np.sqrt) * correlations.apply(np.sign)
        coordinate_correlation_column_names.append(column_name)

    data['correlation_sum'] = data.loc[:, coordinate_correlation_column_names].sum(axis=1)
    data['correlation_diff'] = get_derivative(data['correlation_sum']).abs()

    return data
