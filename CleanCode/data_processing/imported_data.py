from CleanCode.data_processing.imported_data_class import AllData
from CleanCode.data_processing.probes_import.helios import helios_data
from CleanCode.data_processing.probes_import.ulysses import ulysses_data
from CleanCode.data_processing.probes_import.wind import wind_data

from typing import Union


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


def get_classed_data(start_date: str = '27/01/1976', duration: int = 15, start_hour: int = 0, probe: Union[int, str] = 2):
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


if __name__ == '__main__':
    a = get_classed_data()
    print(a)
    print(a.data)
    print(a.probe)
