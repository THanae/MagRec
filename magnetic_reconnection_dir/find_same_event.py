import csv
from datetime import datetime, timedelta
from typing import List

import astropy.units as u
import numpy as np
from heliopy import spice

from data_handler.data_importer.helios_data import HeliosData
from data_handler.orbit_with_spice import kernel_loader, orbit_times_generator, orbit_generator


def get_data_from_csv(events_list_1: str, events_list_2: str):
    event_dates_1 = []
    event_dates_2 = []
    with open(events_list_1) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year, month, day = np.int(row['year']), np.int(row['month']), np.int(row['day'])
            hours, minutes, seconds = np.int(row['hours']), np.int(row['minutes']), np.int(row['seconds'])
            event_dates_1.append(datetime(year, month, day, hours, minutes, seconds))

    with open(events_list_2) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year, month, day = np.int(row['year']), np.int(row['month']), np.int(row['day'])
            hours, minutes, seconds = np.int(row['hours']), np.int(row['minutes']), np.int(row['seconds'])
            event_dates_2.append(datetime(year, month, day, hours, minutes, seconds))

    return event_dates_1, event_dates_2


def find_two_same_events(events_list_1: List[datetime], events_list_2: List[datetime]):
    orbiter1 = kernel_loader(1)
    times = orbit_times_generator(start_date='20/01/1976', end_date='01/10/1979')
    orbit_generator(orbiter1, times)
    orbiter2 = kernel_loader(2)
    orbit_generator(orbiter2, times)
    distance_between_spacecrafts = np.sqrt(
        (orbiter1.x - orbiter2.x) ** 2 + (orbiter1.y - orbiter2.y) ** 2 + (orbiter1.z - orbiter2.z) ** 2)
    _events_list_1 = [datetime(event.year, event.month, event.day) for event in events_list_1]
    _events_list_2 = [datetime(event.year, event.month, event.day) for event in events_list_2]
    allowed_error = 6  # hours
    duration = 10  # over which speed is averaged
    get_events_relations(_events_list=_events_list_1, events_list_probe=events_list_1, events_list_other=events_list_2,
                         distance_between_spacecrafts=distance_between_spacecrafts, probe=1, orbiter=orbiter1,
                         duration=duration, allowed_error=allowed_error)
    get_events_relations(_events_list=_events_list_2, events_list_probe=events_list_2, events_list_other=events_list_1,
                         distance_between_spacecrafts=distance_between_spacecrafts, probe=2, orbiter=orbiter2,
                         duration=duration, allowed_error=allowed_error)


def get_events_relations(_events_list: List[datetime], events_list_probe: List[datetime],
                         events_list_other: List[datetime], distance_between_spacecrafts: list, probe: int,
                         orbiter: spice.Trajectory, duration: int, allowed_error: int):
    print('TEST HELIOS ' + str(probe))
    for n in range(len(events_list_probe)):
        if _events_list[n] in orbiter.times:
            try:
                data_after_event = HeliosData(probe=probe, start_date=(events_list_probe[n]).strftime('%d/%m/%Y'),
                                              duration=duration)
                data_after_event.data.dropna(inplace=True)
                data_after_event.create_processed_column('vp_magnitude')
                speed_after_event = np.mean(data_after_event.data['vp_magnitude'].values)
                data_before_event = HeliosData(probe=probe,
                                               start_date=(events_list_probe[n] - timedelta(hours=duration)).strftime(
                                                   '%d/%m/%Y'), duration=duration)
                data_before_event.data.dropna(inplace=True)
                data_before_event.create_processed_column('vp_magnitude')
                speed_before_event = np.mean(data_after_event.data['vp_magnitude'].values)
                if not np.isnan(speed_after_event) and not np.isnan(speed_before_event):
                    time_taken1 = np.int(
                        (distance_between_spacecrafts[n].to(u.km) / u.km) / speed_after_event)  # dimensionless
                    time_taken2 = np.int(
                        (distance_between_spacecrafts[n].to(u.km) / u.km) / speed_before_event)  # dimensionless
                    date_expected1 = events_list_probe[n] + timedelta(seconds=time_taken1)
                    date_expected2 = events_list_probe[n] - timedelta(seconds=time_taken2)
                    for m in range(len(events_list_other)):
                        if events_list_other[m] - timedelta(hours=allowed_error) <= date_expected1 <= events_list_other[
                                 m] + timedelta(hours=allowed_error):
                            print(date_expected1, events_list_probe[n], events_list_other[m])
                        if events_list_other[m] - timedelta(hours=allowed_error) <= date_expected2 <= events_list_other[
                            m] + timedelta(hours=allowed_error):
                            print(date_expected2, events_list_probe[n], events_list_other[m])
            except RuntimeError:
                print('sorry, no data')
            except RuntimeWarning:
                print('runtime warning')


if __name__ == '__main__':
    helios_1, helios_2 = get_data_from_csv('helios1_magrec.csv', 'helios2_magrec.csv')
    find_two_same_events(helios_1, helios_2)
