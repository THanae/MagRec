from magnetic_reconnection_dir.csv_utils import send_dates_to_csv
from magnetic_reconnection_dir.finder.tests.finder_test import get_possible_reconnections
from magnetic_reconnection_dir.lmn_coordinates import test_reconnection_lmn

probe = 1
parameters = {'sigma_sum': 2.7, 'sigma_diff': 1.9, 'minutes_b': 5}
min_walen = 0.9
max_walen = 1.1
start_date = '13/12/1974'
end_date = '15/08/1984'
radius_to_consider = 1

possible_reconnections = get_possible_reconnections(probe=probe, parameters=parameters, start_time=start_date,
                                                    end_time=end_date, radius=radius_to_consider, data_split='yearly')
lmn_events = test_reconnection_lmn(event_dates=possible_reconnections, probe=probe, minimum_fraction=min_walen,
                                   maximum_fraction=max_walen)

print(lmn_events)
file_name = 'probe'+str(probe) + '_reconnection_events' + '.csv'
send_dates_to_csv(filename=file_name, events_list=lmn_events, probe=probe, add_radius=True)

