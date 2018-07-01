from datetime import timedelta

import pandas as pd
import numpy as np

from data_handler.imported_data import ImportedData
from data_handler.utils.column_processing import get_moving_average, get_derivative, get_outliers
from magnetic_reconnection.finder.base_finder import BaseFinder


class CorrelationFinder(BaseFinder):
    coordinates = ['x', 'y', 'z']

    def __init__(self, outlier_intersection_limit_minutes: int=3):
        super().__init__()
        self.outlier_intersection_limit_minutes = outlier_intersection_limit_minutes


    def find_magnetic_reconnections(self, imported_data: ImportedData):
        self.find_correlations(imported_data.data)
        self.find_outliers(imported_data.data)


        # maybe no need to check if outlier - always seems to be outlier
        # correlation_diff is outlier and
        # (min(correlation_sum left) < -0.5 and max(correlation_sum right) > 0.5) or (max(left) > 0.5 and min(right < 0.5))
        # include actual point in left

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
        print(data.columns.values)
        print(data['correlation_diff'].max())
        return data

    def find_outliers(self, data):
        data['correlation_sum_outliers'] = get_outliers(data['correlation_sum'], standard_deviations=2, ignore_minutes_around=3, reference=0)
        data['correlation_diff_outliers'] = get_outliers(data['correlation_diff'], standard_deviations=1.5)

        outlier_datetimes = []
        # find intersection
        for index, value in data['correlation_diff_outliers'].iteritems():
            index:  pd.Timestamp = index
            interval = timedelta(minutes=self.outlier_intersection_limit_minutes)
            sum_outliers = data.loc[index - interval:index + interval, 'correlation_sum_outliers']
            # ensure there is a positive and a negative value in sum_outliers
            if (sum_outliers > 0).any() and (sum_outliers < 0).any():
                outlier_datetimes.append(index.to_pydatetime())


        # group outliers:

        if outlier_datetimes:
            print(outlier_datetimes, type(outlier_datetimes[0]))
