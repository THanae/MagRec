from functools import partial
import numpy as np
from multiprocessing import Pool
import csv
from typing import List

from CleanCode.parameter_optimisation.mcc_finding import mcc_from_parameters


def find_best_combinations(all_mcc: list, mcc_params: List[dict]):
    """
    Finds the maximum mcc and its corresponding parameters
    :param all_mcc: Matthews Correlation Coefficient
    :param mcc_params: parameters tested in the mcc implementation
    :return:
    """
    # all_mcc = [mcc for mcc in all_mcc if not np.isnan(mcc)]
    maximum_mcc = np.argmax(all_mcc)
    mcc_max = np.max(all_mcc)
    print('The best mcc value is ', mcc_max)
    print(mcc_params[int(maximum_mcc)])


def send_to_csv(name: str, mcc_values: list, mcc_params: List[dict], keys: list):
    """
    Sends the data to a csv file
    :param name: string, name of the file (without the .csv part)
    :param mcc_values: mcc values obtained in the mcc implementation
    :param mcc_params: parameters tested in the mcc implementation
    :param keys: names of the parameters tested in the implementation
    :return:
    """
    with open(name + '.csv', 'w', newline='') as csv_file:
        fieldnames = ['mcc'] + keys
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for n in range(len(mcc_values)):
            mcc_info = {'mcc': mcc_values[n]}
            for key in keys:
                mcc_info[key] = mcc_params[n][key]
            writer.writerow(mcc_info)


if __name__ == '__main__':
    # in np.arange, if args are floats, length of output is ceil((stop - start) / step)
    # eg for (0.7, 1, 0.1), we get 0.7, 0.8, 0.9, 1.0
    parameters = {'xyz': {'sigma_sum': np.arange(19, 31, 2) / 10, 'sigma_diff': np.arange(19, 31, 2) / 10,
                          'minutes_b': [5, 6, 7]},
                  'lmn': {'minimum_walen': np.arange(7, 10, 1) / 10, 'maximum_walen': np.arange(11, 14, 1) / 10}}
    parameters_keys = list(parameters.keys())
    test_args = [
        {'xyz': {'sigma_sum': sigma_s, 'sigma_diff': sigma_d, 'minutes_b': mins_b}, 'lmn': {'minimum_walen': min_wal,
                                                                                            'maximum_walen': max_wal}}
        for sigma_s in parameters['xyz']['sigma_sum'] for sigma_d in
        parameters['xyz']['sigma_diff'] for mins_b in parameters['xyz']['minutes_b'] for min_wal in
        parameters['lmn']['minimum_walen'] for max_wal in parameters['lmn']['maximum_walen']]
    pool = Pool(processes=2)
    with_finder = partial(mcc_from_parameters)
    results = pool.map(with_finder, test_args)
    mcc = [result[0] for result in results]
    params = [result[1] for result in results]

    # send_to_csv('mcc_corr_lmn2', mcc, params, parameters_keys)
    find_best_combinations(mcc, params)
