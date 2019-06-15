from datetime import datetime, timedelta
from typing import List, Union
import numpy as np

from CleanCode.reconnection_detection import get_events_with_params

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


def mcc_from_parameters(mcc_parameters: dict, event_list=events_list) -> List[Union[float, dict]]:
    """
    Returns the mcc with corresponding sigma_sum, sigma_diff and minutes_b
    :param mcc_parameters: dictionary of the parameters to be tested
    :param event_list: list of events from which the mcc is calculated
    :return: list containing the mcc and associated parameters
    """
    f_n, t_n, t_p, f_p = 0, 0, 0, 0
    for event, probe, reconnection_number in event_list:
        print(event, reconnection_number)
        interval = 3
        start_time = event - timedelta(hours=interval / 2)
        reconnection = get_events_with_params(probe, parameters=mcc_parameters,
                                              _start_time=start_time.strftime('%d/%m/%Y'),
                                              _end_time=(start_time + timedelta(hours=interval)).strftime('%d/%m/%Y'),
                                              to_plot=False)
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
