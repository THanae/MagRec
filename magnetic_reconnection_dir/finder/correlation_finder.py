from datetime import timedelta, datetime
from typing import List

import pandas as pd
import numpy as np

from data_handler.data_importer.imported_data import ImportedData
from data_handler.data_importer.helios_data import HeliosData
from data_handler.data_importer.ulysses_data import UlyssesData
from data_handler.utils.column_processing import get_moving_average, get_derivative, get_outliers
from magnetic_reconnection_dir.finder.base_finder import BaseFinder


class CorrelationFinder(BaseFinder):
    coordinates = ['x', 'y', 'z']

    def __init__(self, outlier_intersection_limit_minutes: int = 3):
        super().__init__()
        self.outlier_intersection_limit_minutes = outlier_intersection_limit_minutes

    def find_magnetic_reconnections(self, imported_data: ImportedData, sigma_sum: float = 3, sigma_diff: float = 2.5,
                                    minutes_b: float = 3, minutes: float = 10, nt_test: bool = False) -> List[datetime]:
        """
        Finds possible events by running a series of tests on the data
        :param imported_data: ImportedData
        :param sigma_sum: float
        :param sigma_diff: float
        :param minutes_b: int
        :return:
        """
        self.find_correlations(imported_data.data)
        datetimes_list = self.find_outliers(imported_data.data, sigma_sum=sigma_sum, sigma_diff=sigma_diff,
                                            minutes=minutes)
        datetimes_list = self.b_changes(datetimes_list, imported_data.data, minutes_b=minutes_b)
        if nt_test:
            datetimes_list = self.n_and_t_changes(datetimes_list, imported_data.data)
        return datetimes_list

    def b_changes(self, datetimes_list: list, data: pd.DataFrame, minutes_b: float) -> List[datetime]:
        """
        Tests whether there is a change in the magnitude of b around the considered event
        :param datetimes_list: list of possible events
        :param data: ImportedData
        :param minutes_b: number of minutes around the considered event where b will be considered
        :return: filtered list of possible events
        """
        filtered_datetimes_list: List[datetime] = []
        for _datetime in datetimes_list:
            try:
                interval = timedelta(minutes=minutes_b)
                for coordinate in self.coordinates:
                    b = data['B{}'.format(coordinate)].loc[_datetime - interval:_datetime + interval].dropna()
                    if (b < 0).any() and (b > 0).any() and _datetime in get_average_b2(_datetime,
                                                                                       data['B{}'.format(coordinate)]):
                        filtered_datetimes_list.append(_datetime)
                        break
            except Exception:
                print('Some dates were in invalid format')  # There was a nan

        print('B sign change filter returned: ', filtered_datetimes_list)
        return filtered_datetimes_list

    def n_and_t_changes(self, high_changes_datetime_list: List[datetime], data: pd.DataFrame) -> List[datetime]:
        """
        Checks whether there is a change in density and temperature at a time close to the event time
        :param high_changes_datetime_list: lsit of possible events
        :param data: ImportedData
        :return: filtered list of events
        """
        minutes_nt = 10
        n_and_t_datetime_list: List[datetime] = []
        for _datetime in high_changes_datetime_list:
            interval = timedelta(minutes=minutes_nt)
            n_around = data['n_p'].loc[_datetime - interval:_datetime + interval].dropna()
            t_around = data['Tp_par'].loc[_datetime - interval:_datetime + interval].dropna()
            n_diff = get_derivative(n_around)
            t_diff = get_derivative(t_around)
            n_outliers = get_outliers(n_diff, minutes=minutes_nt, standard_deviations=2, ignore_minutes_around=2,
                                      reference='median')
            t_outliers = get_outliers(t_diff, minutes=minutes_nt, standard_deviations=2, ignore_minutes_around=2,
                                      reference='median')
            if (np.isfinite(n_outliers)).any() and (np.isfinite(t_outliers)).any():
                n_and_t_datetime_list.append(_datetime)
        print('Density and temperature changes filter returned: ', n_and_t_datetime_list)
        return n_and_t_datetime_list

    def find_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Finds all the correlations by multiplying the diffs of b and v together (divided by the time between the data
        points in question)
        These are divided by the standard deviations of b and v to obtain a kind of scaling
        The total correlation is then obtained by summing all correlations
        :param data: ImportedData
        :return:
        """
        coordinate_correlation_column_names = []

        for coordinate in self.coordinates:
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

    def find_outliers(self, data: pd.DataFrame, sigma_sum: float, sigma_diff: float, minutes: float = 10) -> List[
        datetime]:
        """
        Finds all events which have high changes in correlation
        :param data: ImportedData
        :param sigma_sum: sigma faction used in finding the high changes in the total correlations
        :param sigma_diff: sigma faction used in finding the high changes in the difference of the total correlations
        :return: list of possible events
        """
        data['correlation_sum_outliers'] = get_outliers(data['correlation_sum'], standard_deviations=sigma_sum,
                                                        ignore_minutes_around=3, reference=0, minutes=minutes)
        data['correlation_diff_outliers'] = get_outliers(data['correlation_diff'], standard_deviations=sigma_diff,
                                                         minutes=minutes)

        outlier_datetimes = []
        # find intersection
        for index, value in data['correlation_diff_outliers'].iteritems():
            index: pd.Timestamp = index
            interval = timedelta(minutes=self.outlier_intersection_limit_minutes)
            sum_outliers = data.loc[index - interval:index + interval, 'correlation_sum_outliers']
            # ensure there is a positive and a negative value in sum_outliers
            if (sum_outliers > 0).any() and (sum_outliers < 0).any():
                outlier_datetimes.append(index.to_pydatetime())

        n = 0
        grouped_outliers = []
        groups = 0
        while n < len(outlier_datetimes) - 1:
            grouped_outliers.append([])
            grouped_outliers[groups].append(outlier_datetimes[n])
            n += 1
            while (outlier_datetimes[n] - outlier_datetimes[n - 1]).total_seconds() < 130 and n < len(
                    outlier_datetimes) - 1:
                grouped_outliers[groups].append(outlier_datetimes[n])
                n += 1
            groups = groups + 1

        # if grouped_outliers:
        #     print(grouped_outliers, len(grouped_outliers))

        datetimes_list = []
        for group in grouped_outliers:
            # find max correlation_diff_outliers
            maximum_in_group = data.loc[group, 'correlation_diff_outliers']
            datetimes_list.append(maximum_in_group.idxmax())

        print('Outliers check returned: ', datetimes_list)
        return datetimes_list

    def print_reconnection_events(self, reconnection_dates: List[datetime]):
        """
        Prints the events dates in a more readable form
        Can be used later when logs from each tests are not outputted anymore
        :param reconnection_dates: all events that passed the tests
        :return:
        """
        for reconnection_date in reconnection_dates:
            print('event detected at ' + reconnection_date.strftime('%H:%M:%S %d/%m/%Y'))
        if len(reconnection_dates) == 0:
            print("No reconnections were found")


def get_average_b2(_datetime: datetime, data_column) -> List[datetime]:
    """
    Checks whether b really changes magnitude before and after a given event
    :param _datetime: list of possible event
    :param data_column: column that we check
    :param minutes_b: not used - might need to be changed
    :return: nothing if no big change, and the event if there is indeed a change
    """
    high_changes_datetime_list = []
    minutes_b = 10
    interval = timedelta(minutes=minutes_b)
    b_left = data_column.loc[_datetime - interval:_datetime].dropna()
    b_right = data_column.loc[_datetime:_datetime + interval].dropna()

    moving_average_b_left = get_moving_average(b_left, minutes=2)
    moving_average_b_right = get_moving_average(b_right, minutes=2)
    # want to get rid of high middle value that might skew the results
    average_b_left = np.mean(b_left.iloc[:-1].values)
    average_b_right = np.mean(b_right.iloc[1:].values)
    std_b = np.max([(b_left - moving_average_b_left).std(), (b_right - moving_average_b_right).std()])
    if (np.abs(average_b_left - average_b_right) > 2 * std_b or np.isnan(std_b)) and (
            np.sign(average_b_right) != np.sign(average_b_left)):
        high_changes_datetime_list.append(_datetime)
    return high_changes_datetime_list
