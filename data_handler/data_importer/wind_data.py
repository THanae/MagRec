import numpy as np
import pandas as pd
from datetime import timedelta
from heliopy.data import wind

from data_handler.data_importer.imported_data import ImportedData


class WindData(ImportedData):
    def __init__(self, start_date: str = '25/12/1994', duration: int = 15, start_hour: int = 0, probe: str = 'wind'):
        """
        :param start_date: string of 'DD/MM/YYYY'
        :param duration: int in hours
        :param start_hour: int from 0 to 23 indicating starting hour of given start_date
        :param probe: 'ace'
        """
        super().__init__(start_date, duration, start_hour, probe)
        self.data = self.get_imported_data()

        if len(self.data) == 0:
            raise RuntimeWarning('Created AceData object has retrieved no data: {}'.format(self))

    def __repr__(self):
        return '{}: at {:%H:%M %d/%m/%Y} by probe {}. Data has {} entries.'.format(self.__class__.__name__,
                                                                                   self.start_datetime,
                                                                                   self.probe,
                                                                                   len(self.data))

    def get_imported_data(self):
        data_bv = wind.swe_h1(self.start_datetime, self.end_datetime)
        data_bv = data_bv.data  # data_bv was previously a time series
        # data_t = wind.threedp_pm(self.start_datetime, self.end_datetime)
        # data_t = data_t.data  # data_b was previously a time series
        indices = [pd.Timestamp(index).to_pydatetime() for index in data_bv.index.values]
        combined_data = pd.DataFrame(index=indices)
        iteration = 0
        for index in indices:
            interval = 2
            if iteration != 0 and iteration != len(indices) - 1:
                interval = (indices[iteration + 1] - indices[iteration - 1]).total_seconds() / 60
            combined_data.loc[index, 'vp_x'] = data_bv.loc[index, 'Proton_VX_nonlin']
            combined_data.loc[index, 'vp_y'] = data_bv.loc[index, 'Proton_VY_nonlin']
            combined_data.loc[index, 'vp_z'] = data_bv.loc[index, 'Proton_VZ_nonlin']
            combined_data.loc[index, 'n_p'] = data_bv.loc[index, 'Proton_Np_nonlin']
            # for now both temperatures are equal to keep it similar to other classes as no separate data was found
            # combined_data.loc[index, 'Tp_par'] = np.mean(
            #     data_t.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'P_TEMP'])
            # combined_data.loc[index, 'Tp_perp'] = np.mean(
            #     data_t.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'P_TEMP'])
            combined_data.loc[index, 'r_sun'] = 1 - np.sqrt(
                data_bv.loc[index, 'xgse'] ** 2 + data_bv.loc[index, 'ygse'] ** 2 + data_bv.loc[
                    index, 'zgse'] ** 2) * 4.26354E-5  # earth radius to au, 1- because distance initially from earth
            combined_data.loc[index, 'Bx'] = data_bv.loc[index, 'BX']
            combined_data.loc[index, 'By'] = data_bv.loc[index, 'BY']
            combined_data.loc[index, 'Bz'] = data_bv.loc[index, 'BZ']

            iteration += 1

        return combined_data


if __name__ == '__main__':
    x = WindData()
    print(x.data)
