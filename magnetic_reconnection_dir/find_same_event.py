import pprint
from datetime import datetime, timedelta
from typing import List, Union, Any
import astropy.units as u
import numpy as np
from heliopy import spice

from data_handler.data_importer.helios_data import HeliosData
from data_handler.orbit_with_spice import get_orbiter, kernel_loader, orbit_times_generator, orbit_generator
from magnetic_reconnection_dir.csv_utils import get_dates_from_csv, create_events_list_from_csv_files
from magnetic_reconnection_dir.mva_analysis import hybrid_mva


def find_two_same_events(events_list_1: List[datetime], events_list_2: List[datetime]):
    """
    Finds all pairs of related events
    :param events_list_1: list of all events for helios 1
    :param events_list_2: list of all events for helios 2
    :return: possible pair of events with their associated probes
    """
    orbiter1 = get_orbiter(probe=1, start_time='20/01/1976', end_time='01/10/1979')
    orbiter2 = get_orbiter(probe=2, start_time='20/01/1976', end_time='01/10/1979')
    same_events = get_events_relations(events_list_1, events_list_2, orbiter1, orbiter2)
    possible_double_event = []
    for event1, probe1, event2, probe2, dist in same_events:
        print('distance', dist * 6.68459e-9)
        print(event1, event2, probe1, probe2)
        possible_events = find_directions(event1, event2, probe1, probe2)
        if possible_events is not None:
            possible_double_event.append(possible_events)

    return possible_double_event


def get_events_relations(helios1_events: List[datetime], helios2_events: List[datetime],
                         orbiter_helios1: spice.Trajectory, orbiter_helios2: spice.Trajectory,
                         allowed_error: float = 0.2) ->List[List[Union[Union[datetime, int], Any]]]:
    """
    Checks whether two events might be the same by calculated the expected time taken by the solar wind to travel
    between them
    :param helios1_events: events detected by Helios 1
    :param helios2_events: events detected by Helios 2
    :param orbiter_helios1: Helios 1 orbiter
    :param orbiter_helios2: Helios 2 orbiter
    :param allowed_error: allowed percentage error between the theoretical and actual time intervals
    :return: possible pair of events with associated probes and the distance between them
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

            if (end - start).total_seconds() > 864000:  # not the same after 10 days (supposed to die out in 1/2 days)
                continue
            else:
                imported_data_start = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour,
                                                 probe=probe_start, duration=20)
                _end = end - timedelta(hours=20)
                imported_data_end = HeliosData(start_date=_end.strftime('%d/%m/%Y'), start_hour=_end.hour,
                                               probe=probe_end,
                                               duration=20)
                imported_data_start.data.dropna(inplace=True)
                imported_data_end.data.dropna(inplace=True)
                imported_data_start.create_processed_column('vp_magnitude')
                imported_data_end.create_processed_column('vp_magnitude')
                speed_start = np.mean(imported_data_start.data['vp_magnitude'].values)
                speed_end = np.mean(imported_data_end.data['vp_magnitude'].values)
                speed = (speed_start + speed_end) / 2
                speed += (get_alfven(start, probe_start) + get_alfven(end, probe_end)) / 2

                start_position = orbiter_start.times.index(datetime(start.year, start.month, start.day))
                end_position = orbiter_end.times.index(datetime(end.year, end.month, end.day))

                start_x, start_y, start_z = orbiter_start.x[start_position].to(u.km) / u.km, orbiter_start.y[
                    start_position].to(u.km) / u.km, orbiter_start.z[start_position].to(u.km) / u.km
                end_x, end_y, end_z = orbiter_end.x[end_position].to(u.km) / u.km, orbiter_end.y[end_position].to(
                    u.km) / u.km, orbiter_end.z[end_position].to(u.km) / u.km

                probes_separation = np.sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2 + (start_z - end_z) ** 2)
                time_between_events = (end - start).total_seconds()
                expected_time = probes_separation / speed

                if 1 - allowed_error < time_between_events / expected_time < 1 + allowed_error:
                    print(np.abs(orbiter_start.x[start_position] - orbiter_end.x[end_position]),
                          np.abs(orbiter_start.y[start_position] - orbiter_end.y[end_position]),
                          np.abs(orbiter_start.z[start_position] - orbiter_end.z[end_position]))
                    print(time_between_events / expected_time)
                    print('expected ', start + timedelta(seconds=int(expected_time)), ' but got ', end,
                          'starting with probe ', probe_start, start, 'and ending with ', probe_end, end)
                    same_events.append([start, probe_start, end, probe_end, probes_separation])

    return same_events


def find_directions(event1: datetime, event2: datetime, probe1: int, probe2: int, allowed_error=0.2) -> List[
                    Union[datetime, int]]:
    """
    Checks whether the solar wind could have travelled between the two probes by finding the distances between
    the probes in SSE and comparing them to the distance traveled by the solar wind
    :param event1: first event to happen
    :param event2: second event to happen
    :param probe1: probe corresponding to the first event
    :param probe2: probe corresponding to the second event
    :param allowed_error: allowed percentage error in the distance difference between the two obtained distances
    :return: possible pair of events with associated probes
    """
    data1 = HeliosData(start_date=event1.strftime('%d/%m/%Y'), duration=24, probe=probe1)
    data2 = HeliosData(start_date=event2.strftime('%d/%m/%Y'), duration=24, probe=probe2)
    data1.data.dropna(inplace=True)
    data2.data.dropna(inplace=True)

    times = orbit_times_generator(start_date='20/01/1976', end_date='01/10/1979')
    orbiter1_sse = kernel_loader(probe1)
    orbit_generator(orbiter1_sse, times, observing_body='Earth', frame='SSE', probe=probe1)
    orbiter2_sse = kernel_loader(probe2)
    orbit_generator(orbiter2_sse, times, observing_body='Earth', frame='SSE', probe=probe2)

    index1 = orbiter1_sse.times.index(datetime(event1.year, event1.month, event1.day))
    index2 = orbiter2_sse.times.index(datetime(event2.year, event2.month, event2.day))
    x_orb = (orbiter2_sse.x[index2] - orbiter1_sse.x[index1]).to(u.km) / u.km * 6.68459e-9
    y_orb = (orbiter2_sse.y[index2] - orbiter1_sse.y[index1]).to(u.km) / u.km * 6.68459e-9
    z_orb = (orbiter2_sse.z[index2] - orbiter1_sse.z[index1]).to(u.km) / u.km * 6.68459e-9

    time_between_events = (event2 - event1).total_seconds()
    v = np.array(
        [np.mean(data1.data['vp_x'].values), np.mean(data1.data['vp_y'].values), np.mean(data1.data['vp_z'].values)])
    radius = data2.data.loc[event2:event2 + timedelta(minutes=2), 'r_sun'][0] - \
             data1.data.loc[event1:event1 + timedelta(minutes=2), 'r_sun'][0]
    x_dist = time_between_events * v[0] * 6.68459e-9
    y_dist = time_between_events * v[1] * 6.68459e-9
    z_dist = time_between_events * v[2] * 6.68459e-9

    print(np.sqrt(x_orb ** 2 + y_orb ** 2 + z_orb ** 2))
    print(x_orb, y_orb, z_orb)
    print(x_dist, y_dist, z_dist)

    if radius > 0:
        if 1 - allowed_error < np.abs(x_orb / x_dist) < 1 + allowed_error:
            return [event1, event2, probe1, probe2]


def get_alfven(event: datetime, probe: int) -> float:
    """
    Finds the Alfven speed at the event
    :param event: event whose Alfven speed we wznt to find
    :param probe: probe corresponding to the event
    :return: Alfven speed
    """
    interval = timedelta(minutes=5)
    start = event - timedelta(hours=2)
    imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=4, probe=probe)
    imported_data.data.dropna(inplace=True)
    L, M, N = hybrid_mva(event, probe, outside_interval=5, inside_interval=1, mva_interval=10)
    _b = (np.array([np.mean((imported_data.data.loc[event - interval: event + interval, 'Bx']).values),
                    np.mean((imported_data.data.loc[event - interval: event + interval, 'By']).values),
                    np.mean((imported_data.data.loc[event - interval: event + interval, 'Bz']).values)]))
    b_l = np.abs(np.dot(_b, L))
    n = np.mean((imported_data.data.loc[event - interval: event + interval, 'n_p']).values)
    alfven_speed = b_l * 10 ** (-9) / np.sqrt(n * 10 ** 6 * 1.67e-27 * np.pi * 4e-7) * 10 ** (-3)  # we want in km
    return alfven_speed


if __name__ == '__main__':
    helios_1 = create_events_list_from_csv_files([['helios1_magrec2.csv', None], ['helios1mag_rec3.csv', None]])
    helios_2 = create_events_list_from_csv_files([['helios2_magrec2.csv', None], ['helios2mag_rec3.csv', None]])
    a = find_two_same_events(helios_1, helios_2)
    pprint.pprint(a)
