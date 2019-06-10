from typing import Union
from datetime import datetime, timedelta
from pandas import DataFrame


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
