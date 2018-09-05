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
        data_bv = imp.merged(self.probe[4], self.start_datetime, self.end_datetime)
        data_bv = data_bv.data  # data_b was previously a time series
        indices = [pd.Timestamp(index).to_pydatetime() for index in data_bv.index.values]
        combined_data = pd.DataFrame(index=indices)
        for index in indices:
            combined_data.loc[index, 'vp_x'] = data_bv.loc[index, 'vx_mom_gse']
            combined_data.loc[index, 'vp_y'] = data_bv.loc[index, 'vy_mom_gse']
            combined_data.loc[index, 'vp_z'] = data_bv.loc[index, 'vz_mom_gse']
            combined_data.loc[index, 'n_p'] = data_bv.loc[index, 'np_mom']
            # for now both temperatures are equal to keep it similar to other classes as no separate data was found
            combined_data.loc[index, 'Tp_par'] = data_bv.loc[index, 'Tp_mom']
            combined_data.loc[index, 'Tp_perp'] = data_bv.loc[index, 'Tp_mom']
            combined_data.loc[index, 'r_sun'] = 1 - np.sqrt(
                data_bv.loc[index, 'x_gse'] ** 2 + data_bv.loc[index, 'y_gse'] ** 2 + data_bv.loc[
                    index, 'z_gse'] ** 2) * 4.26354E-5  # earth radius to au, 1- because distance initially from earth
            combined_data.loc[index, 'Bx'] = data_bv.loc[index, 'Bx_gse']
            combined_data.loc[index, 'By'] = data_bv.loc[index, 'By_gse']
            combined_data.loc[index, 'Bz'] = data_bv.loc[index, 'Bz_gse']

        return combined_data


if __name__ == '__main__':
    x = ImpData()
    print(x.data)
