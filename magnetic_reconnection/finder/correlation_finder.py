import itertools
from datetime import timedelta, datetime
from typing import List

import pandas as pd
import numpy as np

from data_handler.imported_data import ImportedData
from data_handler.utils.column_processing import get_moving_average, get_derivative, get_outliers
from magnetic_reconnection.finder.base_finder import BaseFinder


class CorrelationFinder(BaseFinder):
    coordinates = ['x', 'y', 'z']

    def __init__(self, outlier_intersection_limit_minutes: int = 3):
        super().__init__()
        self.outlier_intersection_limit_minutes = outlier_intersection_limit_minutes

    def find_magnetic_reconnections(self, imported_data: ImportedData):
        self.find_correlations(imported_data.data)
        datetimes_list = self.find_outliers(imported_data.data)
        datetimes_list = self.b_changes(datetimes_list, imported_data.data)
        datetimes_list = self.get_average_b(datetimes_list, imported_data.data)
        datetimes_list = self.n_and_t_changes(datetimes_list, imported_data.data)
        self.print_reconnection_events(datetimes_list)

    def b_changes(self, datetimes_list, data):
        minutes_b = 2
        filtered_datetimes_list: List[datetime] = []
        try:
            for _datetime in datetimes_list:
                interval = timedelta(minutes=minutes_b)
                for coordinate in self.coordinates:
                    print(_datetime)
                    print(interval)
                    b = data['B{}'.format(coordinate)].loc[_datetime - interval:_datetime + interval].dropna()
                    if (b < 0).any() and (b > 0).any():
                        filtered_datetimes_list.append(_datetime)
                        break
        except Exception:
            print('Sorry wont be possible for these dates')

        # not always good take average and difference in addition to check
        print('B sign change filter returned: ', filtered_datetimes_list)
        return filtered_datetimes_list

    def get_average_b(self, filtered_datetimes_list, data):
        minutes_b = 2
        # get average on left, average on right, take difference, compare to some value (want big value)
        # compare to std on left or right? (min of them)
        # need moving averageeeeeee
        high_changes_datetime_list: List[datetime] = []
        for _datetime in filtered_datetimes_list:
            interval = timedelta(minutes = minutes_b)
            for coordinate in self.coordinates:
                b_left = data['B{}'.format(coordinate)].loc[_datetime - interval:_datetime].dropna()
                b_right = data['B{}'.format(coordinate)].loc[_datetime:_datetime + interval].dropna()
                moving_average_b_left = get_moving_average(b_left, minutes=1)
                moving_average_b_right = get_moving_average(b_right, minutes=1)
                average_b_left = np.mean(b_left.values)
                average_b_right = np.mean(b_right.values)
                # print(b_left.std(), b_right.std())
                std_b = np.min([(b_left - moving_average_b_left).std(), (b_right - moving_average_b_right).std()])
                # if the magnitude difference is bigger than std then there is a bigger chance that it is a reconnection
                # if std is a nan, we just continue and add the date to the list
                if np.abs(average_b_left - average_b_right) > 2*std_b or np.isnan(std_b):
                    high_changes_datetime_list.append(_datetime)
                    break
        print('B magnitude change filter returned ', high_changes_datetime_list)
        return high_changes_datetime_list

    def n_and_t_changes(self, high_changes_datetime_list, data):
        minutes_nt = 10
        n_and_t_datetime_list: List[datetime] = []
        for _datetime in high_changes_datetime_list:
            interval = timedelta(minutes=minutes_nt)
            n_around = data['n_p'].loc[_datetime - interval:_datetime + interval].dropna()
            t_around = data['Tp_par'].loc[_datetime - interval:_datetime + interval].dropna()
            n_diff = get_outliers(n_around, minutes = 10, standard_deviations = 2,ignore_minutes_around = 0, reference='median')
            t_diff = get_outliers(t_around,  minutes = 10, standard_deviations = 2,ignore_minutes_around = 0, reference='median')
            # std_n = (n_around - get_moving_average(n_around)).std()
            # std_t = (t_around - get_moving_average(t_around)).std()
            # if (n_diff.abs() > 2* std_n).any() or (t_diff.abs() > 2*std_t).any():
            if (np.isfinite(n_diff)).any() or (np.isfinite(t_diff)).any():
                n_and_t_datetime_list.append(_datetime)
                break
        print('Density and temperature changes filter returned ', n_and_t_datetime_list)
        return n_and_t_datetime_list

    def find_correlations(self, data: pd.DataFrame):
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
        # print(data.columns.values)
        # print(data['correlation_diff'].max())
        return data

    def find_outliers(self, data) -> List[datetime]:
        data['correlation_sum_outliers'] = get_outliers(data['correlation_sum'], standard_deviations=2,
                                                        ignore_minutes_around=3, reference=0)
        data['correlation_diff_outliers'] = get_outliers(data['correlation_diff'], standard_deviations=1.5)

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

    def print_reconnection_events(self, reconnection_dates):
        for reconnection_date in reconnection_dates:
            print('event detected at ' + reconnection_date.strftime('%H:%M:%S %d/%m/%Y'))
