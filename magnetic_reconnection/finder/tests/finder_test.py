from datetime import timedelta, datetime

from data_handler.distances_with_spice import find_radii, get_time_indices, get_dates, get_data
from data_handler.imported_data import ImportedData
from data_handler.imported_data_plotter import plot_imported_data, DEFAULT_PLOTTED_COLUMNS
from data_handler.orbit_with_spice import kernel_loader, orbit_times_generator, orbit_generator
from magnetic_reconnection.finder.base_finder import BaseFinder
from magnetic_reconnection.finder.correlation_finder import CorrelationFinder
from magnetic_reconnection.finder.tests.known_events import get_known_magnetic_reconnections
from magnetic_reconnection.magnetic_reconnection import MagneticReconnection

import csv
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


def get_test_data(known_event: MagneticReconnection, additional_data_padding_hours: tuple = (1, 2)) -> ImportedData:
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
    test_data = ImportedData(start_date=start_date, start_hour=start_hour, probe=known_event.probe,
                             duration=duration_hours)
    return test_data


def test_finder_with_unknown_events(finder: BaseFinder, imported_data: ImportedData, parameters: list,
                                    plot_reconnections: bool = True) -> list:
    """
    Returns the possible reconnection times as well as the distance from the sun at this time
    :param finder: method to find the reconnections, right now CorrelationFinder
    :param imported_data: ImportedData
    :return:
    """
    interval = 6
    duration = imported_data.duration
    start = imported_data.start_datetime
    probe = imported_data.probe
    reconnections = []
    for n in range(np.int(duration / interval)):
        try:
            data = ImportedData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=interval,
                                probe=probe)
            print(parameters)
            reconnection = finder.find_magnetic_reconnections(data, *parameters)
            if reconnection:
                for event in reconnection:
                    radius = data.data['r_sun'].loc[event]
                    reconnections.append([event, radius])

            if reconnection and plot_reconnections:
                plot_imported_data(data,
                                   DEFAULT_PLOTTED_COLUMNS + [
                                       ('correlation_sum', 'correlation_sum_outliers'),
                                       ('correlation_diff', 'correlation_diff_outliers')]
                                   )
        except Exception:
            print('Exception')
        start = start + timedelta(hours=interval)

    return reconnections


def send_reconnections_to_csv(reconnections_list: list, name: str = 'reconnections'):
    with open(name + '.csv', 'w', newline='') as csv_file:
        fieldnames = ['year', 'month', 'day', 'hours', 'minutes', 'seconds', 'radius']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for reconnection_date, reconnection_radius in reconnections_list:
            year = reconnection_date.year
            month = reconnection_date.month
            day = reconnection_date.day
            hour = reconnection_date.hour
            minutes = reconnection_date.minute
            seconds = reconnection_date.second
            writer.writerow(
                {'year': year, 'month': month, 'day': day, 'hours': hour, 'minutes': minutes, 'seconds': seconds,
                 'radius': reconnection_radius})


def plot_csv(csv_file_name: str, interval: int = 6):
    with open(csv_file_name, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            date = datetime(row['year'], row['month'], row['day'], row['hours'], row['minutes'], 0)
            start_of_plot = date - timedelta(hours=interval / 2)
            imported_data = ImportedData(start_date=start_of_plot.strftime('%d/%m/%Y'), start_hour=start_of_plot.hour,
                                         duration=interval)
            # print('radius: ', row['radius'])
            plot_imported_data(imported_data)


if __name__ == '__main__':
    # test_finder_with_known_events(CorrelationFinder())
    #
    # imported_data = ImportedData(start_date='23/04/1977', start_hour=0, duration=6, probe=2)
    # test_finder_with_unknown_events(CorrelationFinder(), imported_data)
    helios = 2
    orbiter = kernel_loader(helios)
    # times = orbit_times_generator('17/01/1976', '17/01/1979', 1)
    # times = orbit_times_generator('15/12/1974', '15/08/1984', 1)
    times = orbit_times_generator(start_date='27/01/1976', end_date='01/02/1976', interval=1)
    orbit_generator(orbiter, times)
    data = find_radii(orbiter, radius=1)
    time_indices = get_time_indices(data)
    dates = get_dates(orbiter.times, time_indices)
    imported_data_sets = get_data(dates, probe=helios)

    all_reconnections = []

    for n in range(len(imported_data_sets)):
        imported_data = imported_data_sets[n]
        print(imported_data)
        print('duration', imported_data.duration)
        # for no temperature and density check
        # [0.5643202107628984, [2.4641859422774792, 2.7660314936753307, 7.2469545891470277]]
        # [0.56377144873581087, [2.4641859422774792, 1.8348170104492854, 7.2469545891470277]],
        # [0.56377144873581087, [2.4641859422774792, 2.0547187963000382, 7.2469545891470277]],
        # [0.56377144873581087, [2.4641859422774792, 1.8348170104492854, 7.7111085077107582]],
        # [0.56377144873581087, [2.4641859422774792, 2.0547187963000382, 4.8325134468920439]]

        # [0.6415029025857748, [2.6218234455767924, 3.095027734156329, 7.2593589782476995, 0.70195081788266145, 2.2455507200639859]]
        # [0.60733816702295851, [2.6218234455767924, 2.6336660039080351, 7.2593589782476995, 0.62350710758313632, 2.2455507200639859]]
        # [0.60567724060434713, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911, 0.52318189602675891, 1.3497446427829947]]
        # [0.5865557254410011, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911, 0.52318189602675891, 1.3497446427829947]]
        # [0.58208550008719917, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911, 0.52318189602675891, 1.3497446427829947]]
        # [0.57570773308979784, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911, 0.52318189602675891, 1.3497446427829947]]
        # [0.56781364048140137, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911, 0.52318189602675891, 1.3497446427829947]]
        # [0.5661385170722979, [2.6092274374199254, 1.6706268020860184, 6.5797744662938911, 0.50402008003822241, 1.3497446427829947]]
        # [0.5661385170722979, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911, 0.52318189602675891, 1.3497446427829947]]
        # [0.5661385170722979, [2.6218234455767924, 2.6336660039080351, 5.5394039335768328, 0.70195081788266145, 1.2122919887662766]],
        # [0.5657484434338037, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911, 0.52318189602675891, 1.3497446427829947]]
        # [0.56182433421813216, [2.6218234455767924, 2.6336660039080351, 7.2593589782476995, 0.62350710758313632, 2.2455507200639859]]
        # [0.56152597329519949, [2.6092274374199254, 1.6706268020860184, 6.5797744662938911, 0.50402008003822241, 1.3497446427829947]]
        # [0.56147782942977942, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911, 0.52318189602675891, 1.3497446427829947]]

        # for temperature and density check
        # [0.57966713380524337, [2.2123901689753174, 1.1033447004561121, 7.8832162705192079]],
        # [0.57966713380524337, [2.2123901689753174, 1.2000893028353374, 7.8832162705192079]],
        # [0.57966713380524337, [2.2123901689753174, 1.1033447004561121, 5.386013326987813]],
        parameters = {'sigma_sum': 2.4641859422774792, 'sigma_diff': 2.0547187963000382,
                      'minutes_b': 4.8325134468920439}
        params = [parameters[key] for key in list(parameters.keys())]
        reconnection_events = test_finder_with_unknown_events(CorrelationFinder(), imported_data,
                                                              params, plot_reconnections=False)
        if reconnection_events:
            for event in reconnection_events:
                all_reconnections.append(event)

    print(all_reconnections)

    to_csv = False
    if to_csv:
        send_reconnections_to_csv(all_reconnections, 'reconnectionshelios2testdata2')

    # Helios 1 : December 10, 1974 to February 18, 1985
    # BUT the data available from berkeley is from 13 December 1974 to 16 August 1984
    # Helios 2 : January 15, 1976 to December 23, 1979
