import csv
from datetime import datetime, timedelta
from typing import List, Union

import numpy as np

from data_handler.data_importer.helios_data import HeliosData


def get_dates_from_csv(filename: str, probe=None):
    """
    :param filename: name of the file to get the dates from
    :param probe: probe associated with the events, if None the probe is not included in the returned list
    :return:
    """
    events_list = []
    with open(filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year, month, day = np.int(row['year']), np.int(row['month']), np.int(row['day'])
            hours, minutes, seconds = np.int(row['hours']), np.int(row['minutes']), np.int(row['seconds'])
            if probe is not None:
                events_list.append([datetime(year, month, day, hours, minutes, seconds), probe])
            else:
                events_list.append(datetime(year, month, day, hours, minutes, seconds))
    return events_list


def send_dates_to_csv(filename: str, events_list: List[datetime], probe: int, add_radius: bool = True):
    """
    :param filename: name of the output file
    :param events_list: list of events to send to csv
    :param probe: probe corresponding to the events
    :param add_radius: if True, adds the position of the probe at each event
    :return:
    """
    with open(filename + '.csv', 'w', newline='') as csv_file:
        fieldnames = ['year', 'month', 'day', 'hours', 'minutes', 'seconds']
        if add_radius:
            fieldnames.append('radius')
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for reconnection_date in events_list:
            year, month, day = reconnection_date.year, reconnection_date.month, reconnection_date.day
            hour, minutes, seconds = reconnection_date.hour, reconnection_date.minute, reconnection_date.second
            if add_radius:
                start = reconnection_date - timedelta(hours=1)
                imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=2,
                                             probe=probe)
                radius = imported_data.data['r_sun'].loc[
                         reconnection_date - timedelta(minutes=1): reconnection_date + timedelta(minutes=1)][0]
                writer.writerow(
                    {'year': year, 'month': month, 'day': day, 'hours': hour, 'minutes': minutes, 'seconds': seconds,
                     'radius': radius})
            if not add_radius:
                writer.writerow(
                    {'year': year, 'month': month, 'day': day, 'hours': hour, 'minutes': minutes, 'seconds': seconds})


def create_events_list_from_csv_files(files: List[List[Union[str, int]]]):
    """
    Creates list from different events and probes
    :param files: list of lists of files and associated probe
    :return:
    """
    events = []
    for file, probe in files:
        _events = get_dates_from_csv(file, probe)
        for event in _events:
            if event not in events:
                events.append(event)
    return events
