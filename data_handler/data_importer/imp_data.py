from datetime import timedelta

import numpy as np
import pandas as pd
from heliopy.data import imp

from data_handler.data_importer.imported_data import ImportedData


class ImpData(ImportedData):
    def __init__(self, start_date: str = '01/02/1974', duration: int = 15, start_hour: int = 0, probe: str = 'imp_8'):
        """
        :param start_date: string of 'DD/MM/YYYY'
        :param duration: int in hours
        :param start_hour: int from 0 to 23 indicating starting hour of given start_date
        :param probe: imp_ + imp number (from 1 to 8)
        """
        super().__init__(start_date, duration, start_hour, probe)
        self.data = self.get_imported_data()

        if len(self.data) == 0:
            raise RuntimeWarning('Created ImpData object has retrieved no data: {}'.format(self))

    def __repr__(self):
        return '{}: at {:%H:%M %d/%m/%Y} by probe {}. Data has {} entries.'.format(self.__class__.__name__,
                                                                                   self.start_datetime,
                                                                                   self.probe,
                                                                                   len(self.data))

    def get_imported_data(self):
        # only works with imp_8 so far
        data_b = imp.mag15s(self.probe[4], self.start_datetime, self.end_datetime)
        data_v = imp.merged(self.probe[4], self.start_datetime, self.end_datetime)
        data_b = data_b.data  # data_b was previously a time series
        data_v = data_v.data  # data_b was previously a time series
        indices = [pd.Timestamp(index).to_pydatetime() for index in data_v.index.values]
        combined_data = pd.DataFrame(index=indices)
        iteration = 0
        for index in indices:
            interval = 2
            if iteration != 0 and iteration != len(indices) - 1:
                interval = (indices[iteration + 1] - indices[iteration - 1]).total_seconds() / 60
            combined_data.loc[index, 'vp_x'] = data_v.loc[index, 'vx_mom_gse']
            combined_data.loc[index, 'vp_y'] = data_v.loc[index, 'vy_mom_gse']
            combined_data.loc[index, 'vp_z'] = data_v.loc[index, 'vz_mom_gse']
            combined_data.loc[index, 'n_p'] = data_v.loc[index, 'np_mom']
            # for now both temperatures are equal to keep it similar to other classes as no separate data was found
            combined_data.loc[index, 'Tp_par'] = data_v.loc[index, 'Tp_mom']
            combined_data.loc[index, 'Tp_perp'] = data_v.loc[index, 'Tp_mom']
            combined_data.loc[index, 'r_sun'] = 1 - np.sqrt(
                data_v.loc[index, 'x_gse'] ** 2 + data_v.loc[index, 'y_gse'] ** 2 + data_v.loc[
                    index, 'z_gse'] ** 2) * 4.26354E-5  # earth radius to au, 1- because distance initially from earth
            combined_data.loc[index, 'Bx'] = np.mean(
                data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'Bx gse'])
            combined_data.loc[index, 'By'] = np.mean(
                data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'By gse'])
            combined_data.loc[index, 'Bz'] = np.mean(
                data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'Bz gse'])

            iteration += 1
        self.data = combined_data
        return self.data


if __name__ == '__main__':
    x = ImpData()
    print(x.data)
