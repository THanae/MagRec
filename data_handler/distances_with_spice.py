from data_handler.orbit_with_spice import kernel_loader, orbit_times_generator, orbit_generator
import numpy as np
import pandas as pd
from data_handler.data_importer.imported_data import ImportedData
from data_handler.data_importer.helios_data import HeliosData
from data_handler.data_importer.ulysses_data import UlyssesData
from datetime import timedelta
import heliopy.spice as spice
from typing import List


def find_radii(orbiter: spice.Trajectory, radius: float = 0.4):
    """
    Finds all dates at which the radius from the sun is smaller than a given radius
    Much faster than other methods because different way of importing data
    :param radius: radius below which we want the data
    :return: a pandas data frame with the reduced data
    """
    radii = np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2)
    aphelion = np.min(radii)
    perihelion = np.max(radii)
    orbiter_data = pd.DataFrame(
        data={'Times': orbiter.times, 'X': orbiter.x, 'Y': orbiter.y, 'Z': orbiter.z, 'radius': radii})

    reduced = orbiter_data['radius'] < radius
    reduced_data = orbiter_data[reduced]
    return reduced_data


def get_time_indices(reduced_data: pd.DataFrame) -> list:
    """
    We want to make shorter lists of dates that follow each other in order to finally get the data
    :param reduced_data:
    :return:
    """
    completed = False
    m = 0
    n = 0
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
        m = m + 1
    time_indices = []
    for n in range(len(list_of_indices)):
        start = min(list_of_indices[n])
        end = max(list_of_indices[n])
        time_indices.append([start, end])
    return time_indices


def get_dates(orbiter_times: pd.DataFrame, time_indices: list) -> list:
    """
    Function that finds the start and end dates of
    :param reduced_data: data frame, where the times are in the column 'Time'
    :param time_indices: start and end indices of different periods
    :return: start and end dates
    """
    all_dates = []
    for indices in time_indices:
        start_index = indices[0]
        end_index = indices[1]
        start = orbiter_times[start_index]
        end = orbiter_times[end_index]
        all_dates.append([start, end])
    # print(all_dates)
    return all_dates


def get_data(dates: list, probe: int = 2) -> List[ImportedData]:
    """
    Gets the data as ImportedData for the given start and end dates
    Be careful, especially for Helios 1  where a lot of data is missing
    :param dates: list of start and end dates when the spacecraft is at a location smaller than the given radius
    :return: a list of ImportedData
    """
    imported_data = []
    for n in range(len(dates)):
        start = dates[n][0]
        end = dates[n][1]
        delta_t = end - start
        hours = np.int(delta_t.total_seconds() / 3600)
        start_date = start.strftime('%d/%m/%Y')
        try:
            imported_data.append(HeliosData(start_date=start_date, duration=hours, probe=probe))
            # print(imported_data)
        except Exception:
            print('easy method is not working')
            hard_to_get_data = []
            interval = 24
            number_of_loops = np.int(hours/interval)
            for n in range(number_of_loops):
                try:
                    hard_to_get_data.append(HeliosData(start_date=start.strftime('%d/%m/%Y'), duration=interval, probe=probe))
                except Exception:
                    print('not possible to download data between ' + str(start) + ' and ' +str(start+timedelta(hours=interval)))
                start = start + timedelta(hours=interval)

            for n in range(len(hard_to_get_data)):
                imported_data.append(hard_to_get_data[n])
            #
            # grouped_data = []
            # n = 0
            # m = 0
            # while n < len(hard_to_get_data)-1:
            #     print(len(hard_to_get_data))
            #     grouped_data.append([])
            #     grouped = False
            #     while not grouped:
            #         if len(hard_to_get_data) == 1:
            #             grouped_data[m].append(hard_to_get_data[n])
            #             m += 1
            #             n += 1
            #             grouped = True
            #         elif hard_to_get_data[n].end_datetime == hard_to_get_data[n+1].start_datetime:
            #             grouped_data[m].append([hard_to_get_data[n], hard_to_get_data[n+1]])
            #             n = n+1
            #         else:
            #             grouped = True
            #             m += 1
            #             n += 1
            # for groups in grouped_data:
            #     start = groups[0][0].start_datetime
            #     end = groups[-1][1].end_datetime
            #     delta_t = end - start
            #     hours = np.int(delta_t.total_seconds() / 3600)
            #     imported_data.append(ImportedData(start_date=start.strftime('%d/%m/%Y'), duration=hours, probe=probe))

    hours_to_analyse = 0
    for n in range(len(imported_data)):
        a = imported_data[n]
        # print(type(a))
        hours_to_analyse = hours_to_analyse + len(a.data) * 40 / 3600
    print(hours_to_analyse, ' hours to analyse')
    return imported_data


if __name__ == '__main__':
    orbiter = kernel_loader(2)
    times = orbit_times_generator('17/01/1976', '17/01/1979', 1)
    orbit_generator(orbiter, times)
    data = find_radii(orbiter, radius=0.3)
    time_indices = get_time_indices(data)
    dates = get_dates(orbiter.times, time_indices)
    imported_data_sets = get_data(dates)
    # plot_orbit(orbiter, 2)

# Helios 1 : December 10, 1974 to February 18, 1985
# Helios 2 : January 15, 1976 to December 23, 1979
