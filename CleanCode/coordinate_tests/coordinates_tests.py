from datetime import datetime, timedelta
from typing import List
import pandas as pd
import numpy as np
import logging

from CleanCode.coordinate_tests.coordinates_utils import COORDINATES, get_moving_average, get_outliers

logger = logging.getLogger(__name__)


def magnetic_field_tests(date_times_list: list, data: pd.DataFrame, minutes_b: float) -> List[datetime]:
    """
    Determines whether there are changes in the magnetic field sign before and after the data point
    :param date_times_list: dates to test
    :param data: data to test on
    :param minutes_b: number of minutes around the potential event where b will be considered
    :return:
    """
    approved_date_times = []
    for _time in date_times_list:
        try:
            interval = timedelta(minutes=minutes_b)
            for coordinate in COORDINATES:
                b = data['B{}'.format(coordinate)].loc[_time - interval:_time + interval].dropna()
                if (b < 0).any() and (b > 0).any() and average_magnetic_field_tests(_time,
                                                                                    data['B{}'.format(coordinate)]):
                    approved_date_times.append(_time)
                    break
        except TypeError:
            pass  # There was a nan
    logger.debug(f'B sign change filter returned: {approved_date_times}')
    return approved_date_times


def average_magnetic_field_tests(date_time: datetime, data_column: pd.DataFrame, minutes_around: int = 10) -> List[
                                 datetime]:
    """
    Tests whether there are changes in the magnetic field average before and after the data point
    :param date_time: dates to test
    :param data_column: column to test (one of the magnetic field column, but the coordinate is pre-set)
    :param minutes_around: number of minutes around the event which are considered
    :return:
    """
    approved_date_times = []
    interval = timedelta(minutes=minutes_around)
    b_left = data_column.loc[date_time - interval:date_time].dropna()
    b_right = data_column.loc[date_time:date_time + interval].dropna()
    moving_average_b_left = get_moving_average(b_left, minutes=2)
    moving_average_b_right = get_moving_average(b_right, minutes=2)
    # want to get rid of high middle value that might skew the results
    average_b_left = np.mean(b_left.iloc[:-1].values)
    average_b_right = np.mean(b_right.iloc[1:].values)
    std_b = np.max([(b_left - moving_average_b_left).std(), (b_right - moving_average_b_right).std()])
    if (np.abs(average_b_left - average_b_right) > 2 * std_b or np.isnan(std_b)) and (
            np.sign(average_b_right) != np.sign(average_b_left)):
        approved_date_times.append(date_time)
    return approved_date_times


def outliers_test(data: pd.DataFrame, sigma_sum: float, sigma_diff: float, minutes: float = 10) -> List[
                datetime]:
    """
    Checks whether there are changes in the correlation between b and v before and after supposed event
    :param data: data to test
    :param sigma_sum: sigma faction used in finding the high changes in the total (summed) correlations
    :param sigma_diff: sigma faction used in finding high changes in the difference of the total (summed) correlations
    :param minutes: minutes during which the data will be considered for the outliers tests
    :return:
    """
    data['correlation_sum_outliers'] = get_outliers(data['correlation_sum'], standard_deviations=sigma_sum,
                                                    ignore_minutes_around=3, reference=0, minutes=minutes)
    data['correlation_diff_outliers'] = get_outliers(data['correlation_diff'], standard_deviations=sigma_diff,
                                                     minutes=minutes)

    outlier_date_time = []
    for index, value in data['correlation_diff_outliers'].iteritems():
        index: pd.Timestamp = index
        interval = timedelta(minutes=minutes)
        sum_outliers = data.loc[index - interval:index + interval, 'correlation_sum_outliers']
        # ensure there is a positive and a negative value in sum_outliers
        if (sum_outliers > 0).any() and (sum_outliers < 0).any():
            outlier_date_time.append(index.to_pydatetime())

    n, groups = 0, 0
    grouped_outliers = []
    while n < len(outlier_date_time) - 1:
        grouped_outliers.append([])
        grouped_outliers[groups].append(outlier_date_time[n])
        n += 1
        while (outlier_date_time[n] - outlier_date_time[n - 1]).total_seconds() < 130 and n < len(
                outlier_date_time) - 1:
            grouped_outliers[groups].append(outlier_date_time[n])
            n += 1
        groups = groups + 1

    possible_date_times = []
    for group in grouped_outliers:
        maximum_in_group = data.loc[group, 'correlation_diff_outliers']  # find max correlation_diff_outliers
        possible_date_times.append(maximum_in_group.idxmax())

    logger.debug(f'Outliers check returned: {possible_date_times}')
    return possible_date_times
