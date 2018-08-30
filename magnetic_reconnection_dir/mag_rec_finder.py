from magnetic_reconnection_dir.csv_utils import send_dates_to_csv
from magnetic_reconnection_dir.finder.tests.finder_test import get_possible_reconnection_events
from magnetic_reconnection_dir.lmn_coordinates import test_reconnection_lmn

import os


"""
All the detection code can be run from this file. The correlation part is the longest (can last overnight), but the 
LMN part is relatively short (a few minutes for the Helios probes
"""

"""
The following parameters have to be filled in by the parameters.
Probe: 1, 2 or ulysses for now
Parameters: parameters to be used for the correlation tests
Min_walen: minimum fraction of the Alfven speed that the event must have at the exhaust
Max_walen: maximum fraction of the Alfven speed that the event must have at the exhaust
Start_date: start date of the analysis, must be a string
End_date: end date of the analysis, must be a string
Radius_to_consider: maximum radius from the Sun of the events to consider
Noise_when_part1_done: if True, will warn the user when the first part of the program is finished
"""
probe = 1
parameters = {'sigma_sum': 2.7, 'sigma_diff': 1.9, 'minutes_b': 5}
min_walen = 0.9
max_walen = 1.1
start_date = '13/12/1974'
end_date = '15/08/1984'
radius_to_consider = 1
noise_when_part1_done = True

"""
During the first part of the program, changes in correlation are detected by the program
"""
possible_reconnection_events = get_possible_reconnection_events(probe=probe, parameters=parameters, start_time=start_date,
                                                                end_time=end_date, radius=radius_to_consider,
                                                                data_split='yearly')

possible_reconnection_dates = [possible_reconnection[0] for possible_reconnection in possible_reconnection_events]


def beep():
    return os.system("echo '\a'")


"""
If the user wants to be warned when the first part is finished, the program beeps at that point
"""
if noise_when_part1_done:
    beep()

"""
The events are then run though a series of tests in LMN coordinates
"""
lmn_events = test_reconnection_lmn(event_dates=possible_reconnection_dates, probe=probe, minimum_fraction=min_walen,
                                   maximum_fraction=max_walen)

print(lmn_events)
file_name = 'probe'+str(probe) + '_reconnection_events' + '.csv'
send_dates_to_csv(filename=file_name, events_list=lmn_events, probe=probe, add_radius=True)

