from datetime import datetime, timedelta
from typing import Union

import numpy as np
import pandas as pd

from data_handler.utils.column_creator import SUPPORTED_COLUMNS


class ImportedData:
    def __init__(self, start_date: str = '27/01/1976', duration: int = 15, start_hour: int = 0,
                 probe: Union[int, str] = 2):
        """
        :param start_date: string of 'DD/MM/YYYY'
        :param duration: int in hours
        :param start_hour: int from 0 to 23 indicating starting hour of given start_date
        :param probe: 1 for Helios 1, 2 for Helios 2
        """
        self.probe = probe
        self.duration = duration
        self.start_datetime = datetime.strptime(start_date + '/%i' % start_hour, '%d/%m/%Y/%H')
        self.end_datetime = self.start_datetime + timedelta(hours=duration)
        self.data = self.get_imported_data()

        if len(self.data) == 0:
            raise RuntimeWarning('Created ImportedData object has retrieved no data: {}'.format(self))

    def __repr__(self):
        return '{}: at {:%H:%M %d/%m/%Y} by probe {}. Data has {} entries.'.format(self.__class__.__name__,
                                                                                   self.start_datetime,
                                                                                   self.probe,
                                                                                   len(self.data))

    def get_imported_data(self):
        raise NotImplementedError('Need to implement get_imported_data that imports the data for a given spacecraft')

    def create_processed_column(self, column_to_create: str):
        if column_to_create not in SUPPORTED_COLUMNS:
            raise Exception(
                'Column: %s not supported. Supported columns are: %s' % (column_to_create, SUPPORTED_COLUMNS))
        else:
            self.data = SUPPORTED_COLUMNS[column_to_create](self.data)

    def get_moving_average(self, column_name: str, minutes: int = 30):
        """
        Creates column_name_moving_average column in self.data for given column_name
        :param column_name:
        :param minutes:
        :return:
        """
        self.data[column_name + '_moving_average'] = pd.Series(np.zeros_like(self.data[column_name].values),
                                                               index=self.data.index)
        for index, row_data in self.data[column_name].iteritems():
            start_time = index - timedelta(minutes=minutes)
            end_time = index + timedelta(minutes=minutes)
            self.data.loc[index, column_name + '_moving_average'] = np.mean(
                self.data.loc[start_time:end_time, column_name])
            # print(index)
