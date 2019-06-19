from datetime import datetime, timedelta
from typing import Union, List
import logging

from CleanCode.data_processing.imported_data import get_classed_data, get_data_time_basis
from CleanCode.coordinate_tests.coordinates_testing import find_reconnection_list_xyz
from CleanCode.lmn_tests.lmn_testing import lmn_testing
from CleanCode.plots.data_plotter import plot_imported_data
from CleanCode.utils.csv_utils import send_data_to_csv

logger = logging.getLogger(__name__)


def get_events_with_params(_probe: Union[int, str], parameters: dict, _start_time: str, _end_time: str,
                           to_plot: bool = False, to_csv: bool = False) -> List[datetime]:
    """
    Find magnetic reconnection events with the given parameters
    :param _probe: probe to analyse
    :param parameters: parameters to use
    :param _start_time: start time of the analysis
    :param _end_time: end time of the analysis
    :param to_plot: if True, plots the possible events
    :param to_csv: if True, sends potential reconnection events to csv file
    :return:
    """
    # get data
    _start_time = datetime.strptime(_start_time, '%d/%m/%Y')
    _end_time = datetime.strptime(_end_time, '%d/%m/%Y')
    times = []
    while _start_time < _end_time:
        times.append([_start_time, _start_time + timedelta(days=1)])
        _start_time = _start_time + timedelta(days=1)
    imported_data_sets = get_data_time_basis(dates=times, _probe=_probe)

    # find reconnection events with xyz tests
    all_reconnection_events = []
    for n in range(len(imported_data_sets)):
        print(f'Testing data set number {n+1} out of {len(imported_data_sets)}')
        imported_data = imported_data_sets[n]
        logger.debug(f'{imported_data} Duration {imported_data.duration}')
        reconnection_events = find_reconnection_list_xyz(imported_data, **parameters['xyz'])
        if reconnection_events:
            for event in reconnection_events:
                all_reconnection_events.append(event)
    logger.debug(_start_time, _end_time, 'reconnection number: ', str(len(all_reconnection_events)))
    logger.info(f'xyz coordinates test returned {all_reconnection_events}')

    # find events with lmn tests
    lmn_approved_events = []
    duration = 4
    for event in all_reconnection_events:
        _start_time = event - timedelta(hours=duration / 2)
        imported_data = get_classed_data(probe=_probe, start_date=_start_time.strftime('%d/%m/%Y'),
                                         start_hour=_start_time.hour, duration=duration)
        if lmn_testing(imported_data, event, **parameters['lmn']):
            lmn_approved_events.append(event)
            if to_plot:
                plot_imported_data(imported_data, event_date=event)
    logger.info(f'lmn coordinates test returned {lmn_approved_events}')

    # send to csv
    if to_csv:
        send_data_to_csv(f'reconnection_events_{_probe}', lmn_approved_events)
    return lmn_approved_events


if __name__ == '__main__':
    # Can be changed by user if desired
    # logging.basicConfig(level=logging.INFO)
    probe = 2
    # probe = 'ulysses'
    parameters_helios = {'xyz': {'sigma_sum': 2.29, 'sigma_diff': 2.34, 'minutes_b': 6.42, 'minutes': 5.95},
                         'lmn': {'minimum_walen': 0.95, 'maximum_walen': 1.123}}
    # parameters_helios = {'xyz': {'sigma_sum': 2, 'sigma_diff': 2.2, 'minutes_b': 15, 'minutes': 20},
    #                      'lmn': {'minimum_walen': 0.95, 'maximum_walen': 1.123}}
    start_time = '13/12/1978'
    # start_time = '17/02/1992'
    end_time = '17/02/1979'
    # end_time = '17/02/1993'
    plot_events, send_to_csv = False, False
    possible_events = get_events_with_params(_probe=probe, parameters=parameters_helios, _start_time=start_time,
                                             _end_time=end_time, to_plot=plot_events, to_csv=send_to_csv)
    print(possible_events)
