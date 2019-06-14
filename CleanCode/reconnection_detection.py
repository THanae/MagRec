from datetime import datetime, timedelta
import csv

from CleanCode.data_processing.imported_data import get_classed_data, get_data_by_all_means
from CleanCode.coordinate_tests.coordinates_testing import find_reconnection_list_xyz
from CleanCode.lmn_tests.lmn_testing import lmn_testing


# Can be changed by user if desired
probe = 1
parameters_helios = {'sigma_sum': 2.29, 'sigma_diff': 2.34, 'minutes_b': 6.42, 'minutes': 5.95}
start_time = '13/12/1974'
end_time = '17/12/1974'
# end_time = '15/12/1974'


# get data
start_time = datetime.strptime(start_time, '%d/%m/%Y')
end_time = datetime.strptime(end_time, '%d/%m/%Y')
times = []
while start_time < end_time:
    times.append([start_time, start_time + timedelta(days=1)])
    start_time = start_time + timedelta(days=1)
imported_data_sets = get_data_by_all_means(dates=times, _probe=probe)

# find reconnection events with xyz tests
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

# find events with lmn tests
lmn_approved_events = []
duration = 4
for event in all_reconnection_events:
    start_time = event - timedelta(hours=duration / 2)
    imported_data = get_classed_data(probe=probe, start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                                     duration=duration)
    if lmn_testing(imported_data, event, 0.95, 1.123):
        lmn_approved_events.append(event)

print(lmn_approved_events)

# send to csv 
with open(f'reconnection_events_{probe}' + '.csv', 'w', newline='') as csv_file:
        fieldnames = ['year', 'month', 'day', 'hours', 'minutes', 'seconds']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for reconnection_date in lmn_approved_events:
            year, month, day = reconnection_date.year, reconnection_date.month, reconnection_date.day
            hour, minutes, seconds = reconnection_date.hour, reconnection_date.minute, reconnection_date.second
            writer.writerow(
                    {'year': year, 'month': month, 'day': day, 'hours': hour, 'minutes': minutes, 'seconds': seconds})
