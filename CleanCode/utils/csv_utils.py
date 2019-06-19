from datetime import datetime
from typing import List
import numpy as np
import csv


def send_data_to_csv(filename: str, events_list: List[datetime]):
    """
    Sends the found events to a csv file
    :param filename: name of the file to send to (without the .csv)
    :param events_list: list of events
    :return:
    """
    with open(filename + '.csv', 'w', newline='') as csv_file:
        fieldnames = ['year', 'month', 'day', 'hours', 'minutes', 'seconds']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for reconnection_date in events_list:
            year, month, day = reconnection_date.year, reconnection_date.month, reconnection_date.day
            hour, minutes, seconds = reconnection_date.hour, reconnection_date.minute, reconnection_date.second
            writer.writerow(
                    {'year': year, 'month': month, 'day': day, 'hours': hour, 'minutes': minutes, 'seconds': seconds})


def read_data_from_csv(filename: str) -> List[datetime]:
    """
    Reads the events dates from a csv file
    :param filename: name of the file to read from
    :return:
    """
    events_list = []
    with open(filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year, month, day = np.int(row['year']), np.int(row['month']), np.int(row['day'])
            hours, minutes, seconds = np.int(row['hours']), np.int(row['minutes']), np.int(row['seconds'])
            events_list.append(datetime(year, month, day, hours, minutes, seconds))
    return events_list

