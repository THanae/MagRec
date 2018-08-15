from functools import partial
import numpy as np
from multiprocessing import Pool
import csv
from typing import List
from datetime import timedelta, datetime

from data_handler.data_importer.helios_data import HeliosData
from magnetic_reconnection_dir.finder.base_finder import BaseFinder
from magnetic_reconnection_dir.finder.correlation_finder import CorrelationFinder
from magnetic_reconnection_dir.lmn_coordinates import test_reconnection_lmn


# lists [event, probe, number of reconnections]
event_list = [[datetime(1974, 12, 15, 14, 0, 0), 1, 1], [datetime(1974, 12, 15, 20, 0, 0), 1, 1],
              [datetime(1975, 1, 18, 13, 0, 0), 1, 1], [datetime(1975, 2, 7, 1, 0, 0), 1, 1],
              [datetime(1975, 9, 22, 3, 30, 0), 1, 1], [datetime(1975, 12, 19, 21, 0, 0), 1, 1],
              [datetime(1976, 1, 19, 6, 0, 0), 2, 1], [datetime(1976, 1, 27, 7, 0, 0), 2, 1],
              [datetime(1976, 1, 30, 2, 0, 0), 2, 2], [datetime(1976, 3, 4, 9, 0, 0), 2, 1],
              [datetime(1976, 12, 15, 1, 0, 0), 2, 1], [datetime(1977, 4, 5, 22, 0, 0), 2, 1],
              [datetime(1978, 1, 25, 7, 0, 0), 2, 1], [datetime(1978, 2, 26, 4, 0, 0), 2, 1],
              [datetime(1977, 4, 23, 3, 0, 0), 2, 1], [datetime(1977, 12, 17, 1, 0, 0), 1, 1],
              [datetime(1978, 3, 17, 16, 0, 0), 1, 1], [datetime(1979, 6, 21, 2, 0, 0), 1, 1],
              [datetime(1980, 1, 3, 20, 0, 0), 1, 1], [datetime(1980, 1, 16, 14, 0, 0), 1, 1],

              [datetime(1976, 1, 18, 6, 0, 0), 2, 0], [datetime(1976, 2, 2, 7, 0, 0), 2, 0],
              [datetime(1977, 4, 22, 3, 0, 0), 2, 0], [datetime(1976, 2, 4, 7, 0, 0), 2, 0],
              [datetime(1976, 3, 5, 9, 0, 0), 2, 0], [datetime(1976, 12, 16, 1, 0, 0), 2, 0],
              [datetime(1977, 4, 6, 22, 0, 0), 2, 0], [datetime(1977, 12, 19, 1, 0, 0), 2, 0],
              [datetime(1978, 1, 5, 10, 0, 0), 2, 0], [datetime(1974, 12, 17, 14, 0, 0), 1, 0],
              [datetime(1974, 12, 17, 20, 0, 0), 1, 0], [datetime(1975, 1, 19, 13, 0, 0), 1, 0],
              [datetime(1975, 2, 8, 1, 0, 0), 1, 0], [datetime(1975, 9, 24, 3, 30, 0), 1, 0],
              [datetime(1975, 12, 20, 21, 0, 0), 1, 0], [datetime(1977, 12, 18, 1, 0, 0), 1, 0],
              [datetime(1978, 3, 22, 16, 0, 0), 1, 0], [datetime(1976, 12, 1, 2, 0, 0), 1, 0],
              [datetime(1980, 1, 4, 20, 0, 0), 1, 0], [datetime(1980, 1, 18, 14, 0, 0), 1, 0]
              ]

test_data = []  # try some of the best mcc's on the test data and hope for the same kinds of results


def test_with_values(parameters: dict, finder: BaseFinder) -> list:
    """
    Returns the mcc with corresponding sigma_sum, sigma_diff and minutes_b
    :param parameters: dictionary of the parameters to be tested
    :param finder: finder to be used in the tests
    :return:
    """
    f_n, t_n, t_p, f_p = 0, 0, 0, 0
    for event, probe, reconnection_number in event_list:
        interval = 3
        start_time = event - timedelta(hours=interval / 2)
        start_hour = event.hour
        data = HeliosData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_hour, duration=interval,
                          probe=probe)

        # making sure this function can possibly be used with other finders
        # this way we unfold the arguments necessary for the finder, that are fed in the function
        split_of_params = len(parameters) - 2
        list_of_params = [parameters[key] for key in list(parameters.keys())]
        reconnection_corr = finder.find_magnetic_reconnections(data, *list_of_params[:split_of_params])
        reconnection = test_reconnection_lmn(reconnection_corr, probe, *list_of_params[split_of_params:])
        if reconnection_number == 0:
            if len(reconnection) == 0:  # nothing detected, which is good
                t_n += 1
            else:  # too many things detected
                f_p += len(reconnection)
        else:
            if len(reconnection) < reconnection_number:  # not enough detected
                f_n += reconnection_number - len(reconnection)
                t_p += len(reconnection)
            elif len(reconnection) == reconnection_number:  # just enough events detected
                t_p += len(reconnection)
            else:  # more detected than real
                f_p += len(reconnection) - reconnection_number
                t_p += reconnection_number
    mcc = get_mcc(t_p, t_n, f_p, f_n)
    print('MCC', mcc, parameters)
    return [mcc, parameters]


def get_mcc(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    mcc = (true_positives * true_negatives + false_positives * false_negatives) / np.sqrt(
        (true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_negatives) * (
                true_positives + false_positives))
    return mcc


def find_best_combinations(all_mcc: list, params: List[dict]):
    """
    Finds the maximum mcc and its corresponding parameters
    :param all_mcc: Matthews Correlation Coefficient
    :param params: parameters tested in the mcc implementation
    :return:
    """
    # all_mcc = [mcc for mcc in all_mcc if not np.isnan(mcc)]
    maximum_mcc = np.argmax(all_mcc)
    mcc_max = np.max(all_mcc)
    print('The best mcc value is ', mcc_max)
    print(params[maximum_mcc])


def send_to_csv(name: str, mcc: list, params: List[dict], keys: list):
    """
    Sends the data to a csv file
    :param name: string, name of the file (without the .csv part)
    :param mcc: mcc values obtained in the mcc implementation
    :param params: parameters tested in the mcc implementation
    :param keys: names of the parameters tested in the implementation
    :return:
    """
    with open(name + '.csv', 'w', newline='') as csv_file:
        fieldnames = ['mcc'] + keys
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for n in range(len(mcc)):
            mcc_info = {'mcc': mcc[n]}
            for key in keys:
                mcc_info[key] = params[n][key]
            writer.writerow(mcc_info)


if __name__ == '__main__':
    # multiprocessing is faster if your laptop can take it - check max number of processes another laptop could take

    # in np.arange, if args are floats, length of output is ceil((stop - start) / step)
    # eg for np.arange(0.7, 1, 0.1), we get 0.7, 0.8, 0.9, 1.0
    parameters = {'sigma_sum': np.arange(19, 31, 2)/10, 'sigma_diff': np.arange(19, 31, 2)/10, 'minutes_b': [5, 6, 7],
                  'minimum walen': np.arange(7, 10, 1)/10, 'maximum walen': np.arange(11, 14, 1)/10}
    parameters_keys = list(parameters.keys())
    test_args = [{'sigma_sum': sigma_s, 'sigma_diff': sigma_d, 'minutes_b': mins_b, 'minimum walen': min_wal,
                  'maximum walen': max_wal} for sigma_s in parameters['sigma_sum'] for sigma_d in
                 parameters['sigma_diff'] for mins_b in parameters['minutes_b'] for min_wal in
                 parameters['minimum walen'] for max_wal in parameters['maximum walen']]
    pool = Pool(processes=2)
    with_finder = partial(test_with_values, finder=CorrelationFinder())
    results = pool.map(with_finder, test_args)
    mcc = [result[0] for result in results]
    params = [result[1] for result in results]

    send_to_csv('mcc_corr_lmn2', mcc, params, parameters_keys)
    find_best_combinations(mcc, params)

    # MCC 0.737711113563
    # {'sigma_sum': 3.100000000000001, 'sigma_diff': 1.8999999999999999, 'minutes_b': 7,
    # 'minimum walen': 0.99999999999999989, 'maximum walen': 1.2000000000000002}

    # MCC 0.737711113563
    # {'sigma_sum': 2.7000000000000006, 'sigma_diff': 1.8999999999999999, 'minutes_b': 5,
    # 'minimum walen': 0.99999999999999989, 'maximum walen': 1.2000000000000002}
