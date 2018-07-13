from data_handler.imported_data import ImportedData

from datetime import timedelta, datetime
import numpy as np
from magnetic_reconnection.finder.base_finder import BaseFinder
from magnetic_reconnection.finder.correlation_finder import CorrelationFinder
import matplotlib.pyplot as plt
from multiprocessing import Pool
import csv

# list of known event dates, and dates with no reconnections (checked by hand)
# lists [event, probe, number of reconnections]
event_list = [[datetime(1974, 12, 15, 14, 0, 0), 1, 1], [datetime(1974, 12, 15, 20, 0, 0), 1, 1],
              [datetime(1975, 1, 18, 13, 0, 0), 1, 1],
              [datetime(1975, 2, 7, 1, 0, 0), 1, 1], [datetime(1975, 9, 22, 3, 30, 0), 1, 1],
              [datetime(1975, 12, 19, 21, 0, 0), 1, 1],
              [datetime(1976, 1, 19, 6, 0, 0), 2, 1], [datetime(1976, 1, 27, 7, 0, 0), 2, 1],
              [datetime(1976, 1, 30, 2, 0, 0), 2, 2],
              [datetime(1976, 3, 4, 9, 0, 0), 2, 1], [datetime(1976, 12, 15, 1, 0, 0), 2, 1],
              [datetime(1977, 4, 5, 22, 0, 0), 2, 1],
              [datetime(1978, 1, 25, 7, 0, 0), 2, 1], [datetime(1978, 2, 26, 4, 0, 0), 2, 1],
              [datetime(1977, 4, 23, 3, 0, 0), 2, 1],
              [datetime(1977, 12, 17, 1, 0, 0), 1, 1], [datetime(1978, 3, 17, 16, 0, 0), 1, 1],
              [datetime(1979, 6, 21, 2, 0, 0), 1, 1],
              [datetime(1980, 1, 3, 20, 0, 0), 1, 1], [datetime(1980, 1, 16, 14, 0, 0), 1, 1],

              [datetime(1976, 1, 18, 6, 0, 0), 2, 0], [datetime(1976, 2, 2, 7, 0, 0), 2, 0],
              [datetime(1977, 4, 22, 3, 0, 0), 2, 0],
              [datetime(1976, 2, 4, 7, 0, 0), 2, 0], [datetime(1976, 3, 5, 9, 0, 0), 2, 0],
              [datetime(1976, 12, 16, 1, 0, 0), 2, 0],
              [datetime(1977, 4, 6, 22, 0, 0), 2, 0], [datetime(1977, 12, 19, 1, 0, 0), 2, 0],
              [datetime(1978, 1, 5, 10, 0, 0), 2, 0],
              [datetime(1974, 12, 17, 14, 0, 0), 1, 0], [datetime(1974, 12, 17, 20, 0, 0), 1, 0],
              [datetime(1975, 1, 19, 13, 0, 0), 1, 0],
              [datetime(1975, 2, 8, 1, 0, 0), 1, 0], [datetime(1975, 9, 24, 3, 30, 0), 1, 0],
              [datetime(1975, 12, 20, 21, 0, 0), 1, 0],
              [datetime(1977, 12, 18, 1, 0, 0), 1, 0], [datetime(1978, 3, 22, 16, 0, 0), 1, 0],
              [datetime(1976, 12, 1, 2, 0, 0), 1, 0],
              [datetime(1980, 1, 4, 20, 0, 0), 1, 0], [datetime(1980, 1, 18, 14, 0, 0), 1, 0]
              ]


def test_with_values(sigma_and_mins):
    """
    Returns the mcc with corresponding sigma_sum, sigma_diff and minutes_b
    :param sigma_and_mins: tuple of sigma_sum, sigma_diff and minutes_b
    :return:
    """
    sigma_sum = sigma_and_mins[0]
    sigma_diff = sigma_and_mins[1]
    minutes_b = sigma_and_mins[2]
    f_n, t_n, t_p, f_p = 0, 0, 0, 0
    for event, probe, reconnection_number in event_list:
        interval = 3
        start_time = event - timedelta(hours=interval / 2)
        start_hour = event.hour
        data = ImportedData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_hour,
                            duration=interval, probe=probe)
        reconnection = CorrelationFinder.find_magnetic_reconnections(self=CorrelationFinder(),
                                                                     imported_data=data,
                                                                     sigma_sum=sigma_sum,
                                                                     sigma_diff=sigma_diff,
                                                                     minutes_b=minutes_b)
        if reconnection_number == 0:
            if len(reconnection) == 0:
                t_n += 1
            else:
                f_p += len(reconnection)
        else:
            if len(reconnection) < reconnection_number:
                f_n += reconnection_number - len(reconnection)
                t_p += len(reconnection)
            elif len(reconnection) == reconnection_number:
                t_p += len(reconnection)
            else:  # more detected than real
                f_p += len(reconnection) - reconnection_number
                t_p += reconnection_number
    mcc = (t_p * t_n + f_n * f_p) / np.sqrt((t_p + f_p) * (t_p + f_n) * (t_n + f_p) * (t_n + f_n))
    return [mcc, sigma_sum, sigma_diff, minutes_b]


def find_best_combinations(all_mcc, sigma_sum, sigma_diff, minutes_b):
    """
    Finds the maximum mcc and its corresponding parameters
    :param all_mcc: Matthews Correlation Coefficient
    :param sigma_sum: sigma above which the sum of correlation changes is considered significant
    :param sigma_diff: sigma above which the difference in correlation is considered significant
    :param minutes_b: minutes during which the magnetic field is considered around the possible event
    :return:
    """
    maximum_mcc = np.argmax(all_mcc)
    mcc_max = np.max(all_mcc)
    print('The best mcc value is ', mcc_max)
    print(sigma_diff[maximum_mcc], sigma_sum[maximum_mcc], minutes_b[maximum_mcc])


def plot_relationships(mcc, all_sigma_sum, all_sigma_diff, all_minutes_b):
    """
    Plots how the mcc evolves while its different parameters change
    :param mcc: Matthews Correlation Coefficient
    :param all_sigma_sum: sigma above which the sum of correlation changes is considered significant
    :param all_sigma_diff: sigma above which the difference in correlation is considered significant
    :param all_minutes_b: minutes during which the magnetic field is considered around the possible event
    :return:
    """
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(all_sigma_sum, mcc)
    axs[0].set_xlabel('Sigma Sum')
    axs[1].plot(all_sigma_diff, mcc)
    axs[1].set_xlabel('Sigma Diff')
    axs[2].plot(all_minutes_b, mcc)
    axs[2].set_xlabel('Minutes B')
    plt.show()


def send_to_csv(name, mcc, sigma_sum, sigma_diff, minutes_b):
    """
    Sends the data to a csv file
    :param name: string, name of the file (without the .csv part)
    :param sigma_sum: sigma above which the sum of correlation changes is considered significant
    :param sigma_diff: sigma above which the difference in correlation is considered significant
    :param minutes_b: minutes during which the magnetic field is considered around the possible event
    :return:
    """
    with open(name + '.csv', 'w') as csv_file:
        fieldnames = ['mcc', 'sigma_sum', 'sigma_diff', 'minutes_b']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for n in range(len(mcc)):
            writer.writerow(
                {'mcc': mcc[n], 'sigma_sum': sigma_sum[n], 'sigma_diff': sigma_diff[n], 'minutes_b': minutes_b[n]})


if __name__ == '__main__':
    # multiprocessing is faster if your laptop can take it
    # check max number of processes another laptop could take

    sigma_sum = np.arange(1, 4, 0.5)
    sigma_diff = np.arange(1, 4, 0.5)
    minutes_b = [3, 4, 5, 6, 7, 8]
    test_args = [(sigma_s, sigma_d, mins_b) for sigma_s in sigma_sum for sigma_d in sigma_diff for mins_b in minutes_b]
    pool = Pool(processes=2)
    results = pool.map(test_with_values, test_args)
    mcc = [result[0] for result in results]
    sigma_sum = [result[1] for result in results]
    sigma_diff = [result[2] for result in results]
    minutes_b = [result[3] for result in results]

    find_best_combinations(mcc, sigma_sum, sigma_diff, minutes_b)
