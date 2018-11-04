from datetime import datetime, timedelta
from typing import List, Union
import numpy as np

from data_handler.data_importer.data_import import get_probe_data
from magnetic_reconnection_dir.finder.base_finder import BaseFinder
from magnetic_reconnection_dir.lmn_coordinates import test_reconnection_lmn

events_list = (
    [datetime(1974, 12, 15, 14, 0, 0), 1, 1], [datetime(1974, 12, 15, 20, 0, 0), 1, 1],
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
)


def mcc_from_parameters(mcc_parameters: dict, finder: BaseFinder, event_list=events_list) -> List[Union[float, dict]]:
    """
    Returns the mcc with corresponding sigma_sum, sigma_diff and minutes_b
    :param mcc_parameters: dictionary of the parameters to be tested
    :param finder: finder to be used in the tests
    :param event_list: list of events from which the mcc is calculated
    :return: list containing the mcc and associated parameters
    """
    f_n, t_n, t_p, f_p = 0, 0, 0, 0
    for event, probe, reconnection_number in event_list:
        interval = 3
        start_time = event - timedelta(hours=interval / 2)
        start_hour = event.hour
        data = get_probe_data(probe=probe, start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_hour,
                              duration=interval)

        # making sure this function can possibly be used with other finders
        # this way we unfold the arguments necessary for the finder, that are fed in the function
        split_of_params = len(mcc_parameters) - 2
        list_of_params = [mcc_parameters[key] for key in list(mcc_parameters.keys())]
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
    mcc_value = get_mcc(t_p, t_n, f_p, f_n)
    print('MCC', mcc_value, mcc_parameters)
    return [mcc_value, mcc_parameters]


def get_mcc(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    mcc_value = (true_positives * true_negatives + false_positives * false_negatives) / np.sqrt(
        (true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_negatives) * (
                true_positives + false_positives))
    return mcc_value
