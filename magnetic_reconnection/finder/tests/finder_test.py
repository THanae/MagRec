from datetime import timedelta

from data_handler.imported_data import ImportedData
from data_handler.imported_data_plotter import plot_imported_data, DEFAULT_PLOTTED_COLUMNS
from magnetic_reconnection.finder.base_finder import BaseFinder
from magnetic_reconnection.finder.correlation_finder import CorrelationFinder
from magnetic_reconnection.finder.tests.known_events import get_known_magnetic_reconnections
from magnetic_reconnection.magnetic_reconnection import MagneticReconnection


def test_finder_with_known_events(finder: BaseFinder):
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
                               'n_p',
                               'Tp_par',
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


if __name__ == '__main__':
    test_finder_with_known_events(CorrelationFinder())
