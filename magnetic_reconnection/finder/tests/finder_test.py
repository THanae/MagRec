from datetime import timedelta

from data_handler.distances_with_spice import find_radii, get_time_indices, get_dates, get_data
from data_handler.imported_data import ImportedData
from data_handler.imported_data_plotter import plot_imported_data, DEFAULT_PLOTTED_COLUMNS
from data_handler.orbit_with_spice import kernel_loader, orbit_times_generator, orbit_generator
from magnetic_reconnection.finder.base_finder import BaseFinder
from magnetic_reconnection.finder.correlation_finder import CorrelationFinder
from magnetic_reconnection.finder.tests.known_events import get_known_magnetic_reconnections
from magnetic_reconnection.magnetic_reconnection import MagneticReconnection

import numpy as np

def test_finder_with_known_events(finder: BaseFinder):
    """
    Checks whether the finder can detect known events
    :param finder: for now CorrelationFinder
    :return:
    """
    known_events = get_known_magnetic_reconnections()
    for magnetic_reconnection in known_events:
        try:
            test_imported_data = get_test_data(magnetic_reconnection)
        except RuntimeWarning as e:
            print('Excepting error: ' + str(e))
            print('Skipping this event...')
            continue
        print('Created imported_data: ', test_imported_data)
        finder.find_magnetic_reconnections(test_imported_data)
        plot_imported_data(test_imported_data,
                           DEFAULT_PLOTTED_COLUMNS + [
                               # 'correlation_x', 'correlation_y', 'correlation_z',
                               # 'correlation_sum',
                               ('correlation_sum', 'correlation_sum_outliers'),
                               ('correlation_diff', 'correlation_diff_outliers')])

    # test on data


def get_test_data(known_event: MagneticReconnection, additional_data_padding_hours=(1, 2)) -> ImportedData:
    """

    :param known_event: MagneticReconnection with a start_datetime
    :param additional_data_padding_hours: (hours_before, hours_after)
    :return:
    """
    start_datetime = known_event.start_datetime - timedelta(hours=additional_data_padding_hours[0])
    start_date = start_datetime.strftime('%d/%m/%Y')
    start_hour = int(start_datetime.strftime('%H'))
    duration_hours = known_event.duration.seconds // (60 * 60) + sum(additional_data_padding_hours)
    # print(known_event)
    # print({
    #     'start_date': start_date,
    #     'start_hour': start_hour,
    #     'duration': duration_hours
    # })
    test_data = ImportedData(start_date=start_date,
                             start_hour=start_hour,
                             probe=known_event.probe,
                             duration=duration_hours)
    return test_data


def test_finder_with_unknown_events(finder: BaseFinder, imported_data):
    interval = 6
    duration = imported_data.duration
    start = imported_data.start_datetime
    probe = imported_data.probe
    for n in range(np.int(duration/interval)):
        try:
            data = ImportedData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=interval, probe=probe)

            reconnection = finder.find_magnetic_reconnections(data)
            plot = False
            if reconnection or plot:
                plot_imported_data(data,
                               DEFAULT_PLOTTED_COLUMNS  + [
                                   # 'correlation_x', 'correlation_y', 'correlation_z',
                                   # 'correlation_sum',
                                   ('correlation_sum', 'correlation_sum_outliers'),
                                   ('correlation_diff', 'correlation_diff_outliers')]
                                   )
        except Exception:
            print('Exception')
        start = start + timedelta(hours=interval)


if __name__ == '__main__':
    # test_finder_with_known_events(CorrelationFinder())
    #
    # imported_data = ImportedData(start_date='23/04/1977', start_hour=0, duration=6, probe=2)
    # test_finder_with_unknown_events(CorrelationFinder(), imported_data)
    # maybe other event 27/01/1977 at around 2:19

    orbiter = kernel_loader(2)
    times = orbit_times_generator('17/01/1976', '17/01/1979', 1)
    orbit_generator(orbiter, times)
    data = find_radii(orbiter, radius=0.3)
    time_indices = get_time_indices(data)
    dates = get_dates(orbiter.times, time_indices)
    imported_data_sets = get_data(dates)

    for n in range(len(imported_data_sets)):
        imported_data = imported_data_sets[n]
        print('duration', imported_data.duration)
        test_finder_with_unknown_events(CorrelationFinder(), imported_data)
