from typing import List
from datetime import datetime

from CleanCode.coordinate_tests.coordinates_tests import magnetic_field_tests, outliers_test
from CleanCode.coordinate_tests.coordinates_utils import find_correlations


def find_reconnection_list_xyz(imported_data, sigma_sum: float = 3, sigma_diff: float = 2.5,
                               minutes_b: float = 3, minutes: float = 3) -> List[datetime]:
    """
    Finds reconnection events in xyz coordinates (first part of the tests)
    :param imported_data: data to test
    :param sigma_sum: sigma faction used in finding the high changes in the total (summed) correlations
    :param sigma_diff: sigma faction used in finding high changes in the difference of the total (summed) correlations
    :param minutes_b: number of minutes around the potential event where b will be considered
    :param minutes: minutes during which the data will be considered for the outliers tests
    :return:
    """
    find_correlations(imported_data.data)
    possible_events = outliers_test(imported_data.data, sigma_sum=sigma_sum, sigma_diff=sigma_diff,
                                    minutes=minutes)
    if possible_events:
        possible_events = magnetic_field_tests(possible_events, imported_data.data, minutes_b=minutes_b)
    else:
        print('No outlier event found')
        return []
    return possible_events
