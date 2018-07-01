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

    def b_changes(self, datetimes_list, data):
        minutes_b = 3
        filtered_datetimes_list: List[datetime] = []
        for _datetime in datetimes_list:
            interval = timedelta(minutes=minutes_b)
            for coordinate in self.coordinates:
                b = data['B{}'.format(coordinate)].loc[_datetime - interval:_datetime + interval].dropna()
                if (b < 0).any() and (b > 0).any():
                    filtered_datetimes_list.append(_datetime)
                    break
            #
            # bx = np.sign(data['Bx'].loc[
            #              _datetime - :_datetime + timedelta(minutes=minutes_b)].dropna().values)
            # by = np.sign(imported_data.data['By'].loc[
            #              _datetime - timedelta(minutes=minutes_b):_datetime + timedelta(minutes=minutes_b)].dropna().values)
            # bz = np.sign(imported_data.data['Bz'].loc[
            #              _datetime - timedelta(minutes=minutes_b):_datetime + timedelta(minutes=minutes_b)].dropna().values)
            #
            # if (1 in bx and -1 in bx) or (1 in by and -1 in by) or (1 in bz and -1 in bz):
            #     events.append(_datetime)

        print('B sign change filter returned: ', filtered_datetimes_list)
        return filtered_datetimes_list

        # maybe no need to check if outlier - always seems to be outlier
        # correlation_diff is outlier and
        # (min(correlation_sum left) < -0.5 and max(correlation_sum right) > 0.5) or (max(left) > 0.5 and min(right < 0.5))
        # include actual point in left

    def find_correlations(self, data: pd.DataFrame) -> List[pd.datetime]:
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
        print(data.columns.values)
        print(data['correlation_diff'].max())
        return data

    def find_outliers(self, data):
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
