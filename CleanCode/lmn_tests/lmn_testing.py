from datetime import datetime, timedelta
import numpy as np
import logging

from CleanCode.lmn_tests.change_to_lmn import hybrid_mva, get_side_data_v_and_b
from CleanCode.lmn_tests.lmn_tests import b_largest_in_l_direction, multiple_tests, walen_test

logger = logging.getLogger(__name__)


def lmn_testing(imported_data, event_date: datetime, minimum_fraction: float, maximum_fraction: float) -> bool:
    """
    Merges all the LMN tests together for an event. Returns true if the LMN tests are passed
    :param imported_data: data to test on
    :param event_date: possible event date
    :param minimum_fraction: minimum walen fraction
    :param maximum_fraction: maximum walen fraction
    :return:
    """
    # try:
    imported_data.data.dropna(inplace=True)
    probe = imported_data.probe
    if probe == 1 or probe == 2 or probe == 'imp_8' or probe == 'ace' or probe == 'wind':
        mva_interval, outside_interval, inside_interval, min_len = 30, 10, 2, 70
    elif probe == 'ulysses':
        mva_interval, outside_interval, inside_interval, min_len = 60, 30, 10, 5
    else:
        raise NotImplementedError(
            'The probes that have been implemented so far are Helios 1, Helios 2, Imp 8, Ace, Wind and Ulysses')
    if len(imported_data.data) < min_len:
        print('0')
        return False
    L, M, N = hybrid_mva(imported_data, event_date, outside_interval=outside_interval, inside_interval=inside_interval,
                         mva_interval=mva_interval)
    b1, b2, v1, v2 = get_side_data_v_and_b(imported_data, event_date, outside_interval=outside_interval,
                                           inside_interval=inside_interval)
    logger.debug('LMN:', L, M, N, np.dot(L, M), np.dot(L, N), np.dot(M, N), np.dot(np.cross(L, M), N))

    b1_L, b1_M, b1_N = np.dot(L, b1), np.dot(M, b1), np.dot(N, b1)
    b2_L, b2_M, b2_N = np.dot(L, b2), np.dot(M, b2), np.dot(N, b2)
    v1_L, v1_M, v1_N = np.dot(L, v1), np.dot(M, v1), np.dot(N, v1)
    v2_L, v2_M, v2_N = np.dot(L, v2), np.dot(M, v2), np.dot(N, v2)
    b1_changed, b2_changed = np.array([b1_L, b1_M, b1_N]), np.array([b2_L, b2_M, b2_N])
    v1_changed, v2_changed = np.array([v1_L, v1_M, v1_N]), np.array([v2_L, v2_M, v2_N])
    b1_L, b2_L, b1_M, b2_M = b1_changed[0], b2_changed[0], b1_changed[1], b2_changed[1]
    v1_L, v2_L = v1_changed[0], v2_changed[0]

    data_1 = imported_data.data[
             event_date - timedelta(minutes=outside_interval):event_date - timedelta(minutes=inside_interval)]
    data_2 = imported_data.data[
             event_date + timedelta(minutes=inside_interval):event_date + timedelta(minutes=outside_interval)]

    rho_1, rho_2 = np.mean(data_1['n_p'].values), np.mean(data_2['n_p'].values)

    if not b_largest_in_l_direction(b1_L, b2_L, b1_M, b2_M):
        print('1')
        return False
    if not multiple_tests(b1, b2, v1, v2, imported_data, event_date, L):
        print('2')
        return False
    if not walen_test(b1_L, b2_L, v1_L, v2_L, rho_1, rho_2, minimum_fraction, maximum_fraction):
        print('3')
        return False
    return True
    # except ValueError:
    #     print(f'value error for event {event_date}')


# TODO remove test when testing phase is over!!


if __name__ == '__main__':
    from CleanCode.data_processing.imported_data import get_classed_data
    from CleanCode.coordinate_tests.coordinates_testing import find_reconnection_list_xyz

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
    # end_time = '15/12/1974'

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

    print('starting lmn testing')
    lmn_reconnections = []
    duration = 4

    for event in all_reconnection_events:
        start_time = event - timedelta(hours=duration / 2)
        imported_data = get_classed_data(probe=probe, start_date=start_time.strftime('%d/%m/%Y'),
                                       start_hour=start_time.hour, duration=duration)
        if lmn_testing(imported_data, event, 0.998, 1.123):
            lmn_reconnections.append(event)

    print(lmn_reconnections)

