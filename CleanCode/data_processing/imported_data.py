from CleanCode.data_processing.imported_data_class import AllData
from CleanCode.data_processing.probes_import.helios import helios_data
from CleanCode.data_processing.probes_import.ulysses import ulysses_data
from CleanCode.data_processing.probes_import.wind import wind_data

from typing import Union


def probe_import(start_date: str = '27/01/1976', duration: int = 15, start_hour: int = 0, probe: Union[int, str] = 2):
    if probe == 1 or probe == 2:
        return helios_data(start_date, duration, start_hour, probe)
    elif probe == 'ulysses':
        return ulysses_data(start_date, duration, start_hour, probe)
    elif probe == 'wind':
        return wind_data(start_date, duration, start_hour, probe)
    else:
        raise NameError('The program is not currently working with that probe')


def return_class(start_date: str = '27/01/1976', duration: int = 15, start_hour: int = 0, probe: Union[int, str] = 2):
    data = probe_import(start_date, duration, start_hour, probe)
    if len(data) == 0:
        raise RuntimeWarning('Created ImportedData object has retrieved no data: {}'.format(start_date))
    all_data = AllData(start_date, duration, start_hour, probe)
    all_data.data = data
    return all_data


if __name__ == '__main__':
    a = return_class()
    print(a)
    print(a.data)
