from typing import Union

from data_handler.data_importer.helios_data import HeliosData
from data_handler.data_importer.imp_data import ImpData
from data_handler.data_importer.imported_data import ImportedData
from data_handler.data_importer.ulysses_data import UlyssesData


def get_probe_data(probe: Union[int, str], start_date: str, start_hour: int = 0, duration: int = 6) -> ImportedData:
    if probe == 1 or probe == 2:
        imported_data = HeliosData(probe=probe, start_date=start_date, start_hour=start_hour, duration=duration)
    elif probe == 'ulysses':
        imported_data = UlyssesData(start_date=start_date, start_hour=start_hour, duration=duration)
    elif probe == 'imp_8':
        imported_data = ImpData(start_date=start_date, duration=duration, start_hour=start_hour, probe=probe)
    else:
        raise NotImplementedError('This function has only been implemented for Helios 1, Helios 2 and Ulysses so far')
    return imported_data
