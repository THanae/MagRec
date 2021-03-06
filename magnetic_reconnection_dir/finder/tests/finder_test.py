from datetime import timedelta, datetime
from typing import List, Optional, Union
import csv
import numpy as np
import logging

from data_handler.data_importer.data_import import get_probe_data
from data_handler.distances_with_spice import get_imported_data_sets, get_data
from data_handler.data_importer.imported_data import ImportedData
from data_handler.data_importer.helios_data import HeliosData
from data_handler.imported_data_plotter import plot_imported_data, DEFAULT_PLOTTED_COLUMNS
from data_handler.orbit_with_spice import get_orbiter
from magnetic_reconnection_dir.finder.base_finder import BaseFinder
from magnetic_reconnection_dir.finder.correlation_finder import CorrelationFinder
from magnetic_reconnection_dir.finder.tests.known_events import get_known_magnetic_reconnection_events
from magnetic_reconnection_dir.magnetic_reconnection import MagneticReconnection


logger = logging.getLogger(__name__)


def test_finder_with_known_events(finder: BaseFinder):
    """
    Checks whether the finder can detect known events
    :param finder: for now CorrelationFinder
    :return:
    """
    known_events = get_known_magnetic_reconnection_events()
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


def get_test_data(known_event: MagneticReconnection, additional_data_padding_hours: tuple = (1, 2)) -> ImportedData:
    """

    :param known_event: MagneticReconnection with a start_datetime
    :param additional_data_padding_hours: (hours_before, hours_after)
    :return: ImportedData around the magnetic reconnection
    """
    start_datetime = known_event.start_datetime - timedelta(hours=additional_data_padding_hours[0])
    start_date = start_datetime.strftime('%d/%m/%Y')
    start_hour = int(start_datetime.strftime('%H'))
    duration_hours = known_event.duration.seconds // (60 * 60) + sum(additional_data_padding_hours)
    test_data = HeliosData(start_date=start_date, start_hour=start_hour, probe=known_event.probe,
                           duration=duration_hours)
    return test_data


def test_finder_with_unknown_events(finder: BaseFinder, imported_data: ImportedData, parameters: list,
                                    plot_reconnection: bool = True, interval: int = 24) -> List[list]:
    """
    Returns the possible reconnection times as well as the distance from the sun at this time
    :param finder: method to find the reconnection events, right now CorrelationFinder
    :param imported_data: ImportedData
    :param parameters: parameters that will be used in the finder
    :param plot_reconnection: if True, every time a reconnection is detected, it is plotted
    :param interval: interval over which the data is analysed to detect events
    :return: list of possible reconnection events and associated radius from the Sun
    """
    duration = imported_data.duration
    start = imported_data.start_datetime
    probe = imported_data.probe
    reconnection_events = []
    for n in range(np.int(duration / interval)):
        try:
            data = get_probe_data(probe=probe, start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour,
                                  duration=interval)
            reconnection = finder.find_magnetic_reconnections(data, *parameters)
            if reconnection:
                for event in reconnection:
                    radius = data.data['r_sun'].loc[event]
                    reconnection_events.append([event, radius])

            if reconnection and plot_reconnection:
                plot_imported_data(data, DEFAULT_PLOTTED_COLUMNS + [
                    ('correlation_sum', 'correlation_sum_outliers'),
                    ('correlation_diff', 'correlation_diff_outliers')])
        except Exception:
            print('Exception in test_finder_with_unknown_events')
        start = start + timedelta(hours=interval)

    return reconnection_events


def send_reconnection_events_to_csv(reconnection_events_list: list, name: str = 'reconnection_events.csv'):
    """
    :param reconnection_events_list: list of reconnection events dates and radius
    :param name: name of the file to send the events information to
    :return:
    """
    with open(name, 'w', newline='') as csv_file:
        fieldnames = ['year', 'month', 'day', 'hours', 'minutes', 'seconds', 'radius']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for reconnection_date, reconnection_radius in reconnection_events_list:
            year, month, day = reconnection_date.year, reconnection_date.month, reconnection_date.day
            hour, minutes, seconds = reconnection_date.hour, reconnection_date.minute, reconnection_date.second
            writer.writerow(
                {'year': year, 'month': month, 'day': day, 'hours': hour, 'minutes': minutes, 'seconds': seconds,
                 'radius': reconnection_radius})


def plot_csv(csv_file_name: str, interval: int = 6):
    """
    Plots the data from a csv file
    :param csv_file_name: name of the file (including the .csv)
    :param interval: duration of the plot
    :return:
    """
    with open(csv_file_name, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            date = datetime(row['year'], row['month'], row['day'], row['hours'], row['minutes'], 0)
            start_of_plot = date - timedelta(hours=interval / 2)
            imported_data = HeliosData(start_date=start_of_plot.strftime('%d/%m/%Y'), start_hour=start_of_plot.hour,
                                       duration=interval)
            plot_imported_data(imported_data)


def reconnection_detector_with_finder(probe: Union[int, str], parameters: dict, start_time: str, end_time: str,
                                      radius: float) -> List[list]:
    """
    :param probe: 1 or 2 for Helios 1 or 2, imp_8, ulysses, wind or ace
    :param parameters: dictionary of parameters for the finder
    :param start_time: start time of the analysis
    :param end_time:  end time of the analysis
    :param radius: maximum radius to consider
    :return: all possible reconnection events within the given time frame, with associated radius
    """
    try:
        orbiter = get_orbiter(probe=probe, start_time=start_time, end_time=end_time, interval=1)
        imported_data_sets = get_imported_data_sets(probe=probe, orbiter=orbiter, radius=radius)
    except NotImplementedError:  # probe not implemented in spice :(
        print('NO RADIUS ANALYSIS BECAUSE PROBE NOT EXISTING IN SPICE YET')
        start_time = datetime.strptime(start_time, '%d/%m/%Y')
        end_time = datetime.strptime(end_time, '%d/%m/%Y')
        times = []
        while start_time < end_time:
            times.append([start_time, start_time + timedelta(days=1)])
            start_time = start_time + timedelta(days=1)
        imported_data_sets = get_data(dates=times, probe=probe)

    all_reconnection_events = []
    for n in range(len(imported_data_sets)):
        imported_data = imported_data_sets[n]
        print(imported_data)
        print('duration', imported_data.duration)
        params = [parameters[key] for key in list(parameters.keys())]
        reconnection_events = test_finder_with_unknown_events(CorrelationFinder(), imported_data,
                                                              params, plot_reconnection=False)
        if reconnection_events:
            for event in reconnection_events:
                all_reconnection_events.append(event)
    print(start_time, end_time, 'reconnection number: ', str(len(all_reconnection_events)))
    print(all_reconnection_events)
    return all_reconnection_events


def get_possible_reconnection_events(probe: Union[int, str], parameters: dict, start_time: str = '17/12/1974',
                                     end_time: str = '21/12/1975', radius: float = 1, to_csv: bool = False,
                                     data_split: Optional[str] = None) -> List[list]:
    """
    :param probe: 1 or 2 for Helios 1 or 2, imp_8, ulysses, wind or ace
    :param parameters: dictionary of parameters for the finder
    :param start_time: time when the search starts
    :param end_time: time when the search ends
    :param radius: maximum radius to be considered
    :param to_csv: true if we want the data to be sent to csv, false otherwise
    :param data_split: None if we want to download in bulk, 'yearly' otherwise (recommended option for Helios 1 for now)
    :return: list of all possible reconnection events and associated radius
    """
    supported_options = [None, 'yearly']
    all_reconnection_events = []
    if data_split is None:
        all_reconnection_events = reconnection_detector_with_finder(probe, parameters, start_time, end_time,
                                                                    radius=radius)
    elif data_split == 'yearly':
        start_year = datetime.strptime(start_time, '%d/%m/%Y').year
        end_year = datetime.strptime(end_time, '%d/%m/%Y').year
        number_of_years = end_year - start_year
        _start_time = datetime.strptime(start_time, '%d/%m/%Y')
        for n in range(number_of_years):
            _end_time = datetime(start_year + 1, 1, 1, 0, 0)
            reconnection_events = reconnection_detector_with_finder(probe, parameters, _start_time.strftime('%d/%m/%Y'),
                                                                    _end_time.strftime('%d/%m/%Y'), radius=radius)
            for reconnection in reconnection_events:
                all_reconnection_events.append(reconnection)
            start_year += 1
            _start_time = _end_time
    else:
        print('SORRY, THIS OPTION HAS NOT BEEN IMPLEMENTED. THE IMPLEMENTED OPTIONS ARE', supported_options)
    print(start_time, end_time, 'reconnection number: ', str(len(all_reconnection_events)))
    print(all_reconnection_events)
    if to_csv:
        sigma_sum, sigma_dif, minutes_b = parameters['sigma_sum'], parameters['sigma_diff'], parameters['minutes_b']
        sigma_sum, sigma_dif, minutes_b = to_int_str(sigma_sum), to_int_str(sigma_dif), to_int_str(minutes_b)
        send_reconnection_events_to_csv(all_reconnection_events, 'events_probe_' + str(
            probe) + '_' + sigma_sum + '_' + sigma_dif + '_' + minutes_b + '.csv')

    return all_reconnection_events


def to_int_str(param):
    if type(param) == float:
        return str(int(10 * param))
    else:
        return str(param)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # test_finder_with_known_events(CorrelationFinder())
    # imported_data = HeliosData(start_date='23/04/1977', start_hour=0, duration=123, probe=2)
    # test_finder_with_unknown_events(CorrelationFinder(), imported_data,  [2.7, 1.9, 5])

    helios = 1
    # helios = 2
    parameters_helios = {'sigma_sum': 2.29, 'sigma_diff': 2.34, 'minutes_b': 6.42, 'minutes':5.95}
    # parameters_uly = {'sigma_sum': 2.7, 'sigma_diff': 1.9, 'minutes_b': 30, 'minutes': 30}
    parameters_uly = {'sigma_sum': 3, 'sigma_diff': 2.5, 'minutes_b': 30, 'minutes': 30}
    parameters_imp = {'sigma_sum': 3, 'sigma_diff': 2.5, 'minutes_b': 2, 'minutes': 2}
    # analysis_start_date = '17/12/1974'
    # analysis_end_date = '21/12/1975'
    # analysis_start_date = '17/01/1976'
    # analysis_end_date = '17/01/1979'
    analysis_start_date = '13/12/1974'
    analysis_end_date = '15/08/1984'
    radius_to_consider = 1

    get_possible_reconnection_events(probe=1, parameters=parameters_helios, start_time=analysis_start_date,
                                     end_time=analysis_end_date, radius=radius_to_consider, to_csv=True,
                                     data_split='yearly')
    # get_possible_reconnection_events(probe split='yearly')

    # get_possible_reconnection_events(probe=helios, parameters=parameters_helios, start_time=analysis_start_date,
    #                            end_time=analysis_end_date, radius=radius_to_consider, to_csv=True,
    #                            data_split='yearly')

    # get_possible_reconnection_events(probe='imp_8', parameters=parameters_helios, start_time='01/01/1974',
    #                                  end_time='31/12/2000', radius=1, to_csv=True)

    # get_possible_reconnection_events(probe='imp_8', parameters=parameters_imp, start_time='01/01/1974',
    #                                  end_time='31/12/2000', radius=1, to_csv=True)
    # get_possible_reconnection_events(probe='ulysses',
    #                                  parameters={'sigma_sum': 1.5, 'sigma_diff': 1, 'minutes_b': 15, 'minutes': 20},
    #                                  start_time='09/02/1998', end_time='10/02/1998', radius=10, to_csv=False)
    # get_possible_reconnection_events(probe='ulysses',
    #                                  parameters={'sigma_sum': 2, 'sigma_diff': 2.5, 'minutes_b': 15, 'minutes': 20},
    #                                  start_time='01/02/1992', end_time='01/01/2000', radius=10, to_csv=True)

    # get_possible_reconnection_events(probe='ulysses',
    #                                  parameters={'sigma_sum': 2.5, 'sigma_diff': 3, 'minutes_b': 20, 'minutes': 30},
    #                                  start_time='01/02/1992', end_time='11/02/2009', radius=10, to_csv=True)

    # get_possible_reconnection_events(probe='ace', parameters=parameters_helios, start_time='03/04/1998',
    #                                  end_time='25/08/2018', radius=10, to_csv=True)

    # get_possible_reconnection_events(probe='ace', parameters=parameters_helios,
    #                                  start_time='01/01/2017', end_time='01/01/2018', radius=10, to_csv=True)

    # get_possible_reconnection_events(probe='wind', parameters=parameters_helios,
    #                                  start_time='01/01/2007', end_time='01/01/2008', radius=10, to_csv=True)

    # for no temperature and density check
    # [0.6415029025857748, [2.6218234455767924, 3.095027734156329, 7.2593589782476995,
    # 0.70195081788266145, 2.2455507200639859]]
    # [0.60733816702295851, [2.6218234455767924, 2.6336660039080351, 7.2593589782476995,
    # 0.62350710758313632, 2.2455507200639859]]
    # [0.60567724060434713, [2.6067535580543777, 2.6188803433547658, 6.5797744662938911,
    # 0.52318189602675891, 1.3497446427829947]]
    # [0.5643202107628984, [2.4641859422774792, 2.7660314936753307, 7.2469545891470277]]
    # [0.56377144873581087, [2.4641859422774792, 1.8348170104492854, 7.2469545891470277]]

    # for temperature and density check
    # [0.57966713380524337, [2.2123901689753174, 1.1033447004561121, 7.8832162705192079]],
    # [0.57966713380524337, [2.2123901689753174, 1.2000893028353374, 7.8832162705192079]],
    # [0.57966713380524337, [2.2123901689753174, 1.1033447004561121, 5.386013326987813]],

    # Helios 1 : December 10, 1974 to February 18, 1985 BUT berkeley data is from 13 December 1974 to 16 August 1984
    # Helios 2 : January 15, 1976 to December 23, 1979
