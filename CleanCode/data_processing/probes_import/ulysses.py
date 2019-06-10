import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from heliopy.data import ulysses


def ulysses_data(start_date: str = '27/01/1998', duration: int = 15, start_hour: int = 0, probe: str = 'ulysses'):
    start_datetime = datetime.strptime(start_date + '/%i' % start_hour, '%d/%m/%Y/%H')
    end_datetime = start_datetime + timedelta(hours=duration)

    data_b = ulysses.fgm_hires(start_datetime, end_datetime)
    data_v = ulysses.swoops_ions(start_datetime, end_datetime)
    data_b = data_b.data  # fgm_hires now returns a time series (sunpy)
    data_v = data_v.data  # swoops_ions now returns a time series (sunpy)
    indices = [pd.Timestamp(index).to_pydatetime() for index in data_v.index.values]
    combined_data = pd.DataFrame(index=indices)
    iteration = 0
    for index in indices:
        interval = 2
        if iteration != 0 and iteration != len(indices) - 1:
            interval = (indices[iteration + 1] - indices[iteration - 1]).total_seconds() / 60
        combined_data.loc[index, 'vp_x'] = data_v.loc[index, 'v_r']
        combined_data.loc[index, 'vp_y'] = data_v.loc[index, 'v_t']
        combined_data.loc[index, 'vp_z'] = data_v.loc[index, 'v_n']
        combined_data.loc[index, 'n_p'] = data_v.loc[index, 'n_p']
        combined_data.loc[index, 'Tp_par'] = data_v.loc[index, 'T_p_large']
        combined_data.loc[index, 'Tp_perp'] = data_v.loc[index, 'T_p_small']
        combined_data.loc[index, 'r_sun'] = data_v.loc[index, 'r']
        combined_data.loc[index, 'Bx'] = np.mean(
            data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'Bx'])
        combined_data.loc[index, 'By'] = np.mean(
            data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'By'])
        combined_data.loc[index, 'Bz'] = np.mean(
            data_b.loc[index - timedelta(minutes=interval):index + timedelta(minutes=interval), 'Bz'])

        iteration += 1

    return combined_data


if __name__ == '__main__':
    x = ulysses_data()
    print(x)
