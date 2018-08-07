import csv
import pprint
from datetime import datetime, timedelta
from typing import List

import astropy.units as u
import numpy as np
from heliopy import spice

from data_handler.data_importer.helios_data import HeliosData
from data_handler.orbit_with_spice import kernel_loader, orbit_times_generator, orbit_generator
from magnetic_reconnection_dir.csv_utils import get_dates_from_csv


def find_two_same_events(events_list_1: List[datetime], events_list_2: List[datetime]):
    """
    Finds all pairs of related events
    :param events_list_1: list of all events for helios 1
    :param events_list_2: list of all events for helios 2
    :return:
    """
    orbiter1 = kernel_loader(1)
    times = orbit_times_generator(start_date='20/01/1976', end_date='01/10/1979')
    orbit_generator(orbiter1, times)
    orbiter2 = kernel_loader(2)
    orbit_generator(orbiter2, times)
    distance_between_spacecrafts = np.sqrt(
        (orbiter1.x - orbiter2.x) ** 2 + (orbiter1.y - orbiter2.y) ** 2 + (orbiter1.z - orbiter2.z) ** 2)
    _events_list_1 = [datetime(event.year, event.month, event.day) for event in events_list_1]
    _events_list_2 = [datetime(event.year, event.month, event.day) for event in events_list_2]
    allowed_error = 0.02  # percent
    duration = 10  # over which speed is averaged
    same_events_1 = get_events_relations(_events_list=_events_list_1, events_list_probe=events_list_1,
                                         events_list_other=events_list_2,
                                         distance_between_spacecrafts=distance_between_spacecrafts, probe=1,
                                         orbiter=orbiter1, duration=duration, allowed_error=allowed_error)
    # check_speed_correlations(same_events_1, 1, orbiter1, orbiter2)
    same_events_2 = get_events_relations(_events_list=_events_list_2, events_list_probe=events_list_2,
                                         events_list_other=events_list_1,
                                         distance_between_spacecrafts=distance_between_spacecrafts, probe=2,
                                         orbiter=orbiter2, duration=duration, allowed_error=allowed_error)
    # check_speed_correlations(same_events_2, 2, orbiter2, orbiter1)
    same_events = same_events_1 + same_events_2
    potential_same_events = []
    for events in same_events:
        if events not in potential_same_events:
            potential_same_events.append(events)

    return potential_same_events


def get_events_relations(_events_list: List[datetime], events_list_probe: List[datetime],
                         events_list_other: List[datetime], distance_between_spacecrafts: list, probe: int,
                         orbiter: spice.Trajectory, duration: int, allowed_error: float):
    """
    Finds possible pairs of events
    :param _events_list: list of events with hours=0, minutes=0, seconds=0
    :param events_list_probe: list of events for teh probe
    :param events_list_other: list of events for the other probe
    :param distance_between_spacecrafts: distance between the two probes
    :param probe: 1 or 2 for Helios 1 or 2
    :param orbiter: spice.Trajectory
    :param duration: time over which the speed is averaged
    :param allowed_error: maximum percentage difference between theoretical time and real time
    :return:
    """
    same_event = []
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
                        if events_list_other[m] - timedelta(seconds=time_taken1 * allowed_error) <= date_expected1 <= \
                                events_list_other[m] + timedelta(seconds=time_taken1 * allowed_error):
                            print('expected:', date_expected1, 'from probe', str(probe), ':', events_list_probe[n],
                                  'and got:', events_list_other[m])
                            same_event.append([events_list_probe[n], events_list_other[m]])
                        if events_list_other[m] - timedelta(seconds=time_taken2 * allowed_error) <= date_expected2 <= \
                                events_list_other[m] + timedelta(seconds=time_taken2 * allowed_error):
                            print('expected:', date_expected2, 'from probe', str(probe), ':', events_list_probe[n],
                                  'and got:', events_list_other[m])
                            same_event.append([events_list_probe[n], events_list_other[m]])
            except RuntimeError:
                print('sorry, no data')
            except RuntimeWarning:
                print('runtime warning')
    return same_event


def check_speed_correlations(correlated_events: List[List[datetime]], probe, orbiter1, orbiter2):
    min_error = 0.4
    max_error = 1.6
    for correlated_event in correlated_events:
        duration = np.abs((correlated_event[1] - correlated_event[0]).total_seconds())
        crossing1 = datetime(correlated_event[0].year, correlated_event[0].month, correlated_event[0].day)
        crossing2 = datetime(correlated_event[1].year, correlated_event[1].month, correlated_event[1].day)
        c1 = orbiter1.times.index(crossing1)
        c2 = orbiter2.times.index(crossing2)
        orbiter1_x, orbiter1_y, orbiter1_z = orbiter1.x * 1.496e+8 / u.au, orbiter1.y * 1.496e+8 / u.au, orbiter1.z * 1.496e+8 / u.au
        orbiter2_x, orbiter2_y, orbiter2_z = orbiter2.x * 1.496e+8 / u.au, orbiter2.y * 1.496e+8 / u.au, orbiter2.z * 1.496e+8 / u.au

        data = HeliosData(start_date=correlated_event[0].strftime('%d/%m/%Y'), duration=int(duration/3600), probe=probe)
        v_x = np.mean(data.data['vp_x'])
        v_y = np.mean(data.data['vp_y'])
        v_z = np.mean(data.data['vp_z'])
        v = np.sqrt(v_x**2 + v_y**2 + v_z**2)

        print(v_x, v_y, v_z, v)

        x_dist = np.abs(orbiter1_x[c1] - orbiter2_x[c2])
        y_dist = np.abs(orbiter1_y[c1] - orbiter2_y[c2])
        z_dist = np.abs(orbiter1_z[c1] - orbiter2_z[c2])

        _v = np.sqrt((x_dist / duration) ** 2 + (y_dist / duration) ** 2 + (z_dist / duration) ** 2)
        print(_v)

        print(x_dist / v_x, y_dist / v_y, z_dist / v_z, duration)

        if min_error * (x_dist / v_x) <= duration <= max_error * (x_dist / v_x) and min_error * (
                y_dist / v_y) <= duration <= max_error * (y_dist / v_y) and min_error * (
                z_dist / v_z) <= duration <= max_error * (z_dist / v_z):
            print(correlated_event)


if __name__ == '__main__':
    helios_1 = get_dates_from_csv('helios1_magrec2.csv') + get_dates_from_csv('helios1mag_rec3.csv')
    helios_2 = get_dates_from_csv('helios2_magrec2.csv') + get_dates_from_csv('helios2mag_rec3.csv')
    a = find_two_same_events(helios_1, helios_2)
    pprint.pprint(a)
