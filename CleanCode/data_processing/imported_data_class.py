from typing import Union
from datetime import datetime, timedelta
from pandas import DataFrame
import pandas as pd
import numpy as np


class AllData:
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
        self.data = DataFrame()

    def __repr__(self):
        return '{}: at {:%H:%M %d/%m/%Y} by probe {}. Data has {} entries.'.format(self.__class__.__name__,
                                                                                   self.start_datetime,
                                                                                   self.probe,
                                                                                   len(self.data))

    def create_processed_column(self, column_to_create: str):
        if column_to_create not in SUPPORTED_COLUMNS:
            raise Exception(
                'Column: %s not supported. Supported columns are: %s' % (column_to_create, SUPPORTED_COLUMNS))
        else:
            self.data = SUPPORTED_COLUMNS[column_to_create](self.data)


def create_vp_magnitude_column(data: pd.DataFrame)-> pd.DataFrame:
    data['vp_magnitude'] = np.sqrt((data['vp_x']) ** 2 + (data['vp_y']) ** 2 + (data['vp_z']) ** 2)
    return data


def create_b_magnitude_column(data: pd.DataFrame) -> pd.DataFrame:
    data['b_magnitude'] = np.sqrt((data['Bx']) ** 2 + (data['By']) ** 2 + (data['Bz']) ** 2)
    return data


SUPPORTED_COLUMNS = {
    'vp_magnitude': create_vp_magnitude_column,
    'b_magnitude': create_b_magnitude_column,
}