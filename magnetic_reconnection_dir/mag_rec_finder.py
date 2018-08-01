import csv
from datetime import timedelta

from data_handler.data_importer.helios_data import HeliosData
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

with open('probe'+str(probe) +'reconnections' + '.csv', 'w', newline='') as csv_file:
    fieldnames = ['year', 'month', 'day', 'hours', 'minutes', 'seconds', 'radius']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for reconnection_date in lmn_events:
        year, month, day = reconnection_date.year, reconnection_date.month, reconnection_date.day
        hour, minutes, seconds = reconnection_date.hour, reconnection_date.minute, reconnection_date.second
        start = reconnection_date - timedelta(hours=1)
        imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=2,probe=probe)
        radius = imported_data.data['r_sun'].loc[
                 reconnection_date - timedelta(minutes=1): reconnection_date + timedelta(minutes=1)][0]
        writer.writerow(
            {'year': year, 'month': month, 'day': day, 'hours': hour, 'minutes': minutes, 'seconds': seconds,
             'radius': radius})
