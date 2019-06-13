from typing import List
from datetime import datetime, timedelta
import numpy as np


from CleanCode.coordinate_tests.coordinates_tests import magnetic_field_tests, average_magnetic_field_tests, outliers_test
from CleanCode.coordinate_tests.coordinates_utils import find_correlations


def find_reconnection_list_xyz(imported_data, sigma_sum: float = 3, sigma_diff: float = 2.5,
                                    minutes_b: float = 3, minutes: float = 3) -> List[datetime]:
    find_correlations(imported_data.data)
    possible_events = outliers_test(imported_data.data, sigma_sum=sigma_sum, sigma_diff=sigma_diff,
                                                    minutes=minutes)
    if possible_events:
        possible_events = magnetic_field_tests(possible_events, imported_data.data, minutes_b=minutes_b)
    else:
        print('no event')
        return []
    return possible_events


if __name__ == '__main__':
    # TODO remove test below when sure it works
    from CleanCode.data_processing.imported_data import get_classed_data

    def get_data(dates: list, probe: int = 2):
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
                _data = get_classed_data(probe=probe, start_date=start_date, duration=hours)
                imported_data.append(_data)
            except Exception:
                print('Previous method not working, switching to "day-to-day" method')
                hard_to_get_data = []
                interval = 24
                number_of_loops = np.int(hours / interval)
                for loop in range(number_of_loops):
                    try:
                        hard_data = get_classed_data(probe=probe, start_date=start.strftime('%d/%m/%Y'),
                                                   duration=interval)
                        hard_to_get_data.append(hard_data)
                    except Exception:
                        potential_end_time = start + timedelta(hours=interval)
                        print('Not possible to download data between ' + str(start) + ' and ' + str(potential_end_time))
                    start = start + timedelta(hours=interval)

                for loop in range(len(hard_to_get_data)):
                    imported_data.append(hard_to_get_data[n])
        return imported_data


    probe = 1
    parameters_helios = {'sigma_sum': 2.29, 'sigma_diff': 2.34, 'minutes_b': 6.42, 'minutes': 5.95}
    start_time = '13/12/1974'
    end_time = '17/12/1974'

    start_time = datetime.strptime(start_time, '%d/%m/%Y')
    end_time = datetime.strptime(end_time, '%d/%m/%Y')
    times = []
    while start_time < end_time:
        times.append([start_time, start_time + timedelta(days=1)])
        start_time = start_time + timedelta(days=1)
    imported_data_sets = get_data(dates=times, probe=probe)

    all_reconnection_events = []
    for n in range(len(imported_data_sets)):
        imported_data = imported_data_sets[n]
        print(f'{imported_data} Duration {imported_data.duration}')
        params = [parameters_helios[key] for key in list(parameters_helios.keys())]
        reconnection_events = find_reconnection_list_xyz(imported_data, *params)
        if reconnection_events:
            for event in reconnection_events:
                all_reconnection_events.append(event)
    print(start_time, end_time, 'reconnection number: ', str(len(all_reconnection_events)))
    print(all_reconnection_events)
