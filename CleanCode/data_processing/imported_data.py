from typing import Union, List
from datetime import timedelta
import numpy as np

from CleanCode.data_processing.imported_data_class import AllData
from CleanCode.data_processing.probes_import.helios import helios_data
from CleanCode.data_processing.probes_import.ulysses import ulysses_data
from CleanCode.data_processing.probes_import.wind import wind_data


def probe_import(start_date: str = '27/01/1976', duration: int = 15, start_hour: int = 0, probe: Union[int, str] = 2):
    """
    Imports the data of a given probe for a given date
    :param start_date: start time for download
    :param duration: duration of the data to be downloaded
    :param start_hour: hour to start at (defaults at zero)
    :param probe: probe to use
    :return:
    """
    if probe == 1 or probe == 2:
        return helios_data(start_date, duration, start_hour, probe)
    elif probe == 'ulysses':
        return ulysses_data(start_date, duration, start_hour, probe)
    elif probe == 'wind':
        return wind_data(start_date, duration, start_hour, probe)
    else:
        raise NameError('The program is not currently working with that probe')


def get_classed_data(start_date: str = '27/01/1976', duration: int = 15, start_hour: int = 0,
                     probe: Union[int, str] = 2) -> AllData:
    """
    Returns the data in AllData class
    :param start_date: start date of the data
    :param duration: duration fo the data
    :param start_hour: start hour of the data
    :param probe: probe to use
    :return:
    """
    data = probe_import(start_date, duration, start_hour, probe)
    if len(data) == 0:
        raise RuntimeWarning('Created ImportedData object has retrieved no data: {}'.format(start_date))
    all_data = AllData(start_date, duration, start_hour, probe)
    all_data.data = data
    return all_data


def get_data_by_all_means(dates: list, _probe: int = 2) -> List[AllData]:
    """
    Gets the data as AllData for the given start and end dates (a lot of data is missing for Helios 1)
    :param dates: list of start and end dates when the spacecraft is at a location smaller than the given radius
    :param _probe: 1 or 2 for Helios 1 or 2, can also be 'ulysses' or 'imp_8'
    :return: a list of ImportedData for the given dates
    """
    _imported_data = []
    for _n in range(len(dates)):
        start, end = dates[_n][0], dates[_n][1]
        delta_t = end - start
        hours = np.int(delta_t.total_seconds() / 3600)
        start_date = start.strftime('%d/%m/%Y')
        try:
            _data = get_classed_data(probe=_probe, start_date=start_date, duration=hours)
            _imported_data.append(_data)
        except (RuntimeError, RuntimeWarning):
            print('Previous method not working, switching to "day-to-day" method')
            hard_to_get_data = []
            interval = 24
            number_of_loops = np.int(hours / interval)
            for loop in range(number_of_loops):
                try:
                    hard_data = get_classed_data(probe=_probe, start_date=start.strftime('%d/%m/%Y'),
                                                 duration=interval)
                    hard_to_get_data.append(hard_data)
                except (RuntimeError, RuntimeWarning):
                    potential_end_time = start + timedelta(hours=interval)
                    print('Not possible to download data between ' + str(start) + ' and ' + str(potential_end_time))
                start = start + timedelta(hours=interval)

            for loop in range(len(hard_to_get_data)):
                _imported_data.append(hard_to_get_data[_n])
    return _imported_data


if __name__ == '__main__':
    a = get_classed_data()
    print(a)
    print(a.data)
    print(a.probe)
