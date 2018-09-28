import numpy as np
import pandas as pd
from datetime import timedelta
from heliopy.data import ace

from data_handler.data_importer.imported_data import ImportedData


class AceData(ImportedData):
    def __init__(self, start_date: str = '01/01/2001', duration: int = 1, start_hour: int = 0, probe: str = 'ace'):
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
        data_b = ace.mfi_h0(self.start_datetime, self.end_datetime)
        data_b = data_b.data  # data_b was previously a time series
        data_v = ace.swe_h0(self.start_datetime, self.end_datetime)
        data_v = data_v.data  # data_b was previously a time series
        indices = [pd.Timestamp(index).to_pydatetime() for index in data_v.index.values]
        combined_data = pd.DataFrame(index=indices)
        iteration = 0
        for index in indices:
            interval = 2
            if iteration != 0 and iteration != len(indices) - 1:
                interval = (indices[iteration + 1] - indices[iteration - 1]).total_seconds() / 60
            combined_data.loc[index, 'vp_x'] = data_v.loc[index, 'V_GSE_0']
            combined_data.loc[index, 'vp_y'] = data_v.loc[index, 'V_GSE_1']
            combined_data.loc[index, 'vp_z'] = data_v.loc[index, 'V_GSE_2']
            combined_data.loc[index, 'n_p'] = data_v.loc[index, 'Np']
            # for now both temperatures are equal to keep it similar to other classes as no separate data was found
            combined_data.loc[index, 'Tp_par'] = data_v.loc[index, 'Tpr']
            combined_data.loc[index, 'Tp_perp'] = data_v.loc[index, 'Tpr']
            combined_data.loc[index, 'r_sun'] = 1 - np.sqrt(
                data_v.loc[index, 'SC_pos_GSE_0'] ** 2 + data_v.loc[index, 'SC_pos_GSE_1'] ** 2 + data_v.loc[
                    index, 'SC_pos_GSE_2'] ** 2) * 6.68459e-9  # km to au, 1- because distance initially from earth
            combined_data.loc[index, 'Bx'] = np.mean(
                data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'BGSEc_0'])
            combined_data.loc[index, 'By'] = np.mean(
                data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'BGSEc_1'])
            combined_data.loc[index, 'Bz'] = np.mean(
                data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'BGSEc_2'])

            iteration += 1

        return combined_data


if __name__ == '__main__':
    x = AceData()
    print(x.data.dropna())
