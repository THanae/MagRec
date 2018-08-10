import pprint
from datetime import datetime, timedelta
from typing import List

import astropy.units as u
import numpy as np
from heliopy import spice

from data_handler.data_importer.helios_data import HeliosData
from data_handler.orbit_with_spice import get_orbiter
from magnetic_reconnection_dir.csv_utils import get_dates_from_csv
from magnetic_reconnection_dir.mva_analysis import get_b, mva, get_side_data, hybrid_mva


def find_two_same_events(events_list_1: List[datetime], events_list_2: List[datetime]):
    """
    Finds all pairs of related events
    :param events_list_1: list of all events for helios 1
    :param events_list_2: list of all events for helios 2
    :return:
    """
    orbiter1 = get_orbiter(probe=1, start_time='20/01/1976', end_time='01/10/1979')
    orbiter2 = get_orbiter(probe=2, start_time='20/01/1976', end_time='01/10/1979')
    same_events = get_events_relations(events_list_1, events_list_2, orbiter1, orbiter2)
    same_events = check_radial(same_events)

    return same_events


def get_events_relations(helios1_events: List[datetime], helios2_events: List[datetime],
                         orbiter_helios1: spice.Trajectory, orbiter_helios2: spice.Trajectory,
                         allowed_error: float = 0.01):
    """
    Checks whether two events might be the same by calculated the expected time taken by the solar wind to travel
    between them
    :param helios1_events: events detected by Helios 1
    :param helios2_events: events detected by Helios 2
    :param orbiter_helios1: Helios 1 orbiter
    :param orbiter_helios2: Hlelios 2 orbiter
    :param allowed_error: allowed percentage error between the theoretical and actual time intervals
    :return:
    """
    same_events = []
    _helios1_events = []
    for event in helios1_events:
        if datetime(1976, 1, 20) < event < datetime(1979, 10, 1):
            _helios1_events.append(event)
    helios1_events = _helios1_events
    for event_helios1 in helios1_events:
        for event_helios2 in helios2_events:
            if event_helios1 < event_helios2:
                start, end = event_helios1, event_helios2
                probe_start, probe_end = 1, 2
                orbiter_start, orbiter_end = orbiter_helios1, orbiter_helios2
            else:
                start, end = event_helios2, event_helios1
                probe_start, probe_end = 2, 1
                orbiter_start, orbiter_end = orbiter_helios2, orbiter_helios1
            imported_data_start = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour,
                                             probe=probe_start, duration=10)
            _end = end - timedelta(hours=10)
            imported_data_end = HeliosData(start_date=_end.strftime('%d/%m/%Y'), start_hour=_end.hour, probe=probe_end,
                                           duration=10)
            imported_data_start.data.dropna(inplace=True)
            imported_data_end.data.dropna(inplace=True)
            imported_data_start.create_processed_column('vp_magnitude')
            imported_data_end.create_processed_column('vp_magnitude')
            speed_start = np.mean(imported_data_start.data['vp_magnitude'].values)
            speed_end = np.mean(imported_data_end.data['vp_magnitude'].values)
            speed = (speed_start + speed_end) / 2


            start_position = orbiter_start.times.index(datetime(start.year, start.month, start.day))
            end_position = orbiter_end.times.index(datetime(end.year, end.month, end.day))

            print(np.abs(orbiter_start.x[start_position] - orbiter_end.x[end_position]),
                  np.abs(orbiter_start.y[start_position] - orbiter_end.y[end_position]),
                  np.abs(orbiter_start.z[start_position] - orbiter_end.z[end_position]))

            start_x, start_y, start_z = orbiter_start.x[start_position].to(u.km) / u.km, orbiter_start.y[
                start_position].to(u.km) / u.km, orbiter_start.z[start_position].to(u.km) / u.km

            end_x, end_y, end_z = orbiter_end.x[end_position].to(u.km) / u.km, orbiter_end.y[end_position].to(
                u.km) / u.km, orbiter_end.z[end_position].to(u.km) / u.km

            distance_between_spacecrafts = np.sqrt(
                (start_x - end_x) ** 2 + (start_y - end_y) ** 2 + (start_z - end_z) ** 2)

            time_between_events = (end - start).total_seconds()
            expected_time = distance_between_spacecrafts / speed

            if 1 - allowed_error < time_between_events / expected_time < 1 + allowed_error:
                print(time_between_events / expected_time)
                print('expected ', start + timedelta(seconds=int(expected_time)), ' but got ', end,
                      'starting with probe ', probe_start, start, 'and ending with ', probe_end, end)
                same_events.append([start, probe_start, end, probe_end, distance_between_spacecrafts])

    return same_events


def check_radial(same_events: List[list]):
    """
    The solar wind travels more in the radial direction than in the other directions.
    Hence we are expecting the distance between the spacecrafts to be mostly radial
    :param same_events: list of possible same events with the distance between them
    :return:
    """
    possible_same_events = []
    for event1, probe1, event2, probe2, distance_between_events in same_events:
        start = event1 - timedelta(hours=1)
        imported_data1 = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, probe=probe1,
                                    duration=2)
        imported_data1.data.dropna(inplace=True)
        start = event2 - timedelta(hours=1)
        imported_data2 = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, probe=probe2,
                                    duration=2)
        imported_data2.data.dropna(inplace=True)
        radius1 = np.mean(
            imported_data1.data['r_sun'].loc[event1 - timedelta(minutes=5): event1 + timedelta(minutes=5)].values)
        radius2 = np.mean(
            imported_data2.data['r_sun'].loc[event2 - timedelta(minutes=5): event2 + timedelta(minutes=5)].values)
        radial_distance = radius2 - radius1
        distance_between_events = (distance_between_events * u.km).to(u.au) / u.au
        print(radial_distance, distance_between_events, radial_distance / distance_between_events)
        x_distance = (event2 - event1).total_seconds() * np.mean(imported_data1.data['vp_x'].values)
        print(radial_distance, x_distance)
        if radial_distance < 0:
            print('no')  # solar wind travels away from the sun
        else:
            if radial_distance/distance_between_events > 0.2:
                print(radial_distance, distance_between_events, radial_distance / distance_between_events)
                possible_same_events.append([event1, probe1, event2, probe2])
    return possible_same_events


if __name__ == '__main__':
    helios_1 = get_dates_from_csv('helios1_magrec2.csv')
    for dates in get_dates_from_csv('helios1mag_rec3.csv'):
        if dates not in helios_1:
            helios_1.append(dates)
    helios_2 = get_dates_from_csv('helios2_magrec2.csv')
    for dates in get_dates_from_csv('helios2mag_rec3.csv'):
        if dates not in helios_2:
            helios_2.append(dates)
    a = find_two_same_events(helios_1, helios_2)
    pprint.pprint(a)
