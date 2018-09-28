import numpy as np
import pandas as pd
from datetime import timedelta
import heliopy.spice as spice
from typing import List

from data_handler.data_importer.data_import import get_probe_data
from data_handler.data_importer.imported_data import ImportedData
from data_handler.orbit_with_spice import get_orbiter


def find_radii(orbiter: spice.Trajectory, radius: float = 0.4) ->pd.DataFrame:
    """
    Finds all dates at which the radius from the sun is smaller than a given radius
    :param orbiter: spice orbiter of the probe to be analysed
    :param radius: radius below which we want the data
    :return: a pandas data frame with the reduced data
    """
    radii = np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2)
    orbiter_data = pd.DataFrame(data={'times': orbiter.times, 'radius': radii})
    reduced = orbiter_data['radius'] < radius
    reduced_data = orbiter_data[reduced]
    if len(reduced_data) < 2:
        raise IndexError('The data is too small')
    return reduced_data


def get_time_indices(reduced_data: pd.DataFrame) -> List[list]:
    """
    Finds the indices of start and end times when the probe is within a given radius
    :param reduced_data: data that has a distance to the sun less than a given radius
    :return: the time indices of the start and end dates of the intervals
    """
    completed = False
    m, n = 0, 0
    list_of_indices = [[]]
    while not completed:
        if m == len(reduced_data.index) - 2:
            completed = True
        if reduced_data.index[m] == reduced_data.index[m + 1] - 1:
            list_of_indices[n].append(reduced_data.index[m])
            list_of_indices[n].append(reduced_data.index[m + 1])
        else:
            n = n + 1
            list_of_indices.append([])
        m += 1
    time_indices = []
    for n in range(len(list_of_indices)):
        start = min(list_of_indices[n])
        end = max(list_of_indices[n])
        time_indices.append([start, end])
    return time_indices


def get_dates(orbiter_times: pd.DataFrame, time_indices: list) -> List[list]:
    """
    Finds the start and end dates of the intervals
    :param orbiter_times: data frame with the times available in the orbiter
    :param time_indices: start and end indices of different periods
    :return: start and end dates of the intervals
    """
    all_dates = []
    for indices in time_indices:
        start_index, end_index = indices[0], indices[1]
        start, end = orbiter_times[start_index], orbiter_times[end_index]
        all_dates.append([start, end])
    return all_dates


def get_data(dates: list, probe: int = 2) -> List[ImportedData]:
    """
    Gets the data as ImportedData for the given start and end dates (a lot of data is missing for Helios 1)
    :param dates: list of start and end dates when the spacecraft is at a location smaller than the given radius
    :param probe: 1 or 2 for Helios 1 or 2, can also be 'ulysses' or 'imp_8'
    :return: a list of ImportedData for the given dates
    """
    imported_data = []
    for n in range(len(dates)):
        start, end = dates[n][0], dates[n][1]
        delta_t = end - start
        hours = np.int(delta_t.total_seconds() / 3600)
        start_date = start.strftime('%d/%m/%Y')
        try:
            _data = get_probe_data(probe=probe, start_date=start_date, duration=hours)
            imported_data.append(_data)
        except Exception:
            print('Previous method not working, switching to "day-to-day" method')
            hard_to_get_data = []
            interval = 24
            number_of_loops = np.int(hours/interval)
            for loop in range(number_of_loops):
                try:
                    hard_data = get_probe_data(probe=probe, start_date=start.strftime('%d/%m/%Y'), duration=interval)
                    hard_to_get_data.append(hard_data)
                except Exception:
                    potential_end_time = start+timedelta(hours=interval)
                    print('Not possible to download data between ' + str(start) + ' and ' + str(potential_end_time))
                start = start + timedelta(hours=interval)

            for loop in range(len(hard_to_get_data)):
                imported_data.append(hard_to_get_data[n])
    return imported_data


def get_imported_data_sets(probe, orbiter: spice.Trajectory, radius: float) ->List[ImportedData]:
    """
    Finds the imported data sets that correspond to a given radius
    :param probe: probe to consider
    :param orbiter: orbiter of the given probe
    :param radius: radius to consider
    :return: list of ImportedData with a radius smaller than a given radius
    """
    data = find_radii(orbiter, radius=radius)
    time_indices = get_time_indices(data)
    dates = get_dates(orbiter.times, time_indices)
    imported_data_sets = get_data(dates, probe=probe)
    return imported_data_sets


if __name__ == '__main__':
    spacecraft_orbiter = get_orbiter(probe=2, start_time='17/01/1976', end_time='17/01/1979', interval=1)
    data_sets = get_imported_data_sets(probe=2, orbiter=spacecraft_orbiter, radius=0.3)
    print(data_sets)

# Helios 1 : December 10, 1974 to February 18, 1985
# Helios 2 : January 15, 1976 to December 23, 1979
