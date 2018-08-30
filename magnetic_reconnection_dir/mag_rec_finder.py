from typing import Union
import os

from magnetic_reconnection_dir.csv_utils import send_dates_to_csv
from magnetic_reconnection_dir.finder.tests.finder_test import get_possible_reconnection_events
from magnetic_reconnection_dir.lmn_coordinates import test_reconnection_lmn


def df_magnetic_reconnection_events(probe: Union[int, str], parameters: dict, min_walen: float, max_walen: float,
                                    start_date: str, end_date: str, radius_to_consider: float,
                                    noise_when_part1_done: bool, noise_when_part2_done: bool):
    """
    Stands for detect and find magnetic reconnection events
    Sends all possible events for a given probe between given times to a csv file
    All the detection code can be run from this file. The correlation part is the longest (can last overnight), but the
    LMN part is relatively short (a few minutes for the Helios probes)
    :param probe: 1, 2 or ulysses for now
    :param parameters: parameters to be used for the correlation tests
    :param min_walen: minimum fraction of the Alfven speed that the event must have at the exhaust
    :param max_walen: maximum fraction of the Alfven speed that the event must have at the exhaust
    :param start_date: start date of the analysis, must be a string
    :param end_date: end date of the analysis, must be a string
    :param radius_to_consider: maximum radius from the Sun of the events to consider
    :param noise_when_part1_done: if True, will warn the user when the first part of the program is finished
    :param noise_when_part2_done: if True, warns the user when the program has finished running
    :return:
    """

    # During the part 1, changes in correlation are detected
    possible_reconnection_events = get_possible_reconnection_events(probe=probe, parameters=parameters,
                                                                    start_time=start_date, end_time=end_date,
                                                                    radius=radius_to_consider, data_split='yearly')
    possible_reconnection_dates = [possible_reconnection[0] for possible_reconnection in possible_reconnection_events]

    # The user is warned when part 1 is done
    if noise_when_part1_done:
        beep()

    # The events are then run though a series of tests in LMN coordinates
    lmn_events = test_reconnection_lmn(event_dates=possible_reconnection_dates, probe=probe, minimum_fraction=min_walen,
                                       maximum_fraction=max_walen)
    print(lmn_events)

    # the possible dates are sent to a csv file
    file_name = 'probe' + str(probe) + '_reconnection_events' + '.csv'
    send_dates_to_csv(filename=file_name, events_list=lmn_events, probe=probe, add_radius=True)

    if noise_when_part2_done:
        for loop in range(5):
            beep()


def beep():
    return os.system("echo '\a'")


if __name__ == '__main__':
    space_probe = 1
    probe_parameters = {'sigma_sum': 2.7, 'sigma_diff': 1.9, 'minutes_b': 5, 'minutes': 3}
    probe_min_walen = 0.9
    probe_max_walen = 1.1
    probe_start_date = '13/12/1974'
    probe_end_date = '15/08/1984'
    probe_radius_to_consider = 1
    noise1 = True
    noise2 = True
    df_magnetic_reconnection_events(probe=space_probe, parameters=probe_parameters, min_walen=probe_min_walen,
                                    max_walen=probe_max_walen, start_date=probe_start_date, end_date=probe_end_date,
                                    radius_to_consider=probe_radius_to_consider, noise_when_part1_done=noise1,
                                    noise_when_part2_done=noise2)
