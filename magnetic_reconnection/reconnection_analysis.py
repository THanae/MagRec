import csv
import os
import pprint
import numpy as np
from typing import List
from datetime import datetime, timedelta

from data_handler.imported_data import ImportedData
# from magnetic_reconnection.lmn_coordinates import hybrid
from data_handler.orbit_with_spice import kernel_loader, orbit_times_generator, orbit_generator


def distances_stats(events_list: List[datetime], probe: int, only_stats: bool = True) -> dict:
    """
    :param events_list:
    :param probe: 1 or 2 for Helios 1 or 2
    :return: number of reconnections at given distances from the sun
    """
    times_and_radii = {'less than 0.3 au': [], '0.3 to 0.4 au': [], '0.4 to 0.5 au': [], '0.5 to 0.6 au': [],
                       '0.6 to 0.7 au': [], '0.7 to 0.8 au': [], '0.8 to 0.9 au': [], 'above 0.9 au': []}
    for event in events_list:
        start_time = event
        try:
            imported_data = ImportedData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                                         duration=1,
                                         probe=probe)
            radius = imported_data.data['r_sun'].loc[event]
        except Exception:
            radius = np.mean(imported_data.data['r_sun'].values)
        if radius < 0.3:
            times_and_radii['less than 0.3 au'].append([event, radius])
        elif radius < 0.4:
            times_and_radii['0.3 to 0.4 au'].append([event, radius])
        elif radius < 0.5:
            times_and_radii['0.4 to 0.5 au'].append([event, radius])
        elif radius < 0.6:
            times_and_radii['0.5 to 0.6 au'].append([event, radius])
        elif radius < 0.7:
            times_and_radii['0.6 to 0.7 au'].append([event, radius])
        elif radius < 0.8:
            times_and_radii['0.7 to 0.8 au'].append([event, radius])
        elif radius < 0.9:
            times_and_radii['0.8 to 0.9 au'].append([event, radius])
        else:
            times_and_radii['above 0.9 au'].append([event, radius])

    for key in times_and_radii.keys():
        if not only_stats:
            times_and_radii[key].append(str(len(times_and_radii[key])))
        else:
            times_and_radii[key] = (str(len(times_and_radii[key])))
    times_and_radii['total number of reconnections'] = len(events_list)
    pprint.pprint(times_and_radii)
    return times_and_radii


def time_stats(events_list: List[datetime], stats_mode: bool = True) -> dict:
    times = {}
    for event in events_list:
        if str(event.year) not in times.keys():
            times[str(event.year)] = []
            times[str(event.year)].append(event)
        else:
            times[str(event.year)].append(event)
    for key in times.keys():
        if not stats_mode:
            times[key].append(str(len(times[key])))
        else:
            times[key] = str(len(times[key]))
    times['total number of reconnections'] = len(events_list)

    pprint.pprint(times)
    return times


def time_spent_at_distances(probe: int, start_date: str, end_date: str, accuracy: float = 0.5) -> dict:
    orbiter = kernel_loader(probe)
    times = orbit_times_generator(start_date, end_date, interval=accuracy)
    orbit_generator(orbiter, times)
    radii = np.array(np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2))
    time_spent = {}
    time_analysed = datetime.strptime(end_date, '%d/%m/%Y') - datetime.strptime(start_date, '%d/%m/%Y')
    time_analysed = int(time_analysed.total_seconds()/(3600*24))
    time_spent['total time'] = len(radii[np.all([radii < 1.2], axis=0)]) * filecount(probe)/time_analysed
    time_spent['less than 0.3 au'] = len(radii[np.all([radii < 0.3], axis=0)]) / time_spent['total time']
    time_spent['0.3 to 0.4 au'] = len(radii[np.all([radii >= 0.3, radii < 0.4], axis=0)]) / time_spent['total time']
    time_spent['0.4 to 0.5 au'] = len(radii[np.all([radii >= 0.4, radii < 0.5], axis=0)]) / time_spent['total time']
    time_spent['0.5 to 0.6 au'] = len(radii[np.all([radii >= 0.5, radii < 0.6], axis=0)]) / time_spent['total time']
    time_spent['0.6 to 0.7 au'] = len(radii[np.all([radii >= 0.6, radii < 0.7], axis=0)]) / time_spent['total time']
    time_spent['0.7 to 0.8 au'] = len(radii[np.all([radii >= 0.7, radii < 0.8], axis=0)]) / time_spent['total time']
    time_spent['0.8 to 0.9 au'] = len(radii[np.all([radii >= 0.8, radii < 0.9], axis=0)]) / time_spent['total time']
    time_spent['above 0.9 au'] = len(radii[np.all([radii >= 0.9], axis=0)]) / time_spent['total time']

    pprint.pprint(time_spent)
    return time_spent


def time_spent_at_date(probe: int, start_date: str, end_date: str, accuracy: float = 0.5) -> dict:
    orbiter = kernel_loader(probe)
    times = orbit_times_generator(start_date, end_date, interval=accuracy)
    orbit_generator(orbiter, times)
    time_spent = {}
    for time in times:
        if str(time.year) not in time_spent.keys():
            time_spent[str(time.year)] = 1
        else:
            time_spent[str(time.year)] += 1

    for key in time_spent.keys():
        time_spent[key] = (time_spent[key] / len(times)) * filecount(probe, int(key))/365
    time_spent['total time'] = len(times)

    pprint.pprint(time_spent)
    return time_spent


def analyse_dates(events_list: List[datetime], probe: int, start_date: str, end_date: str):
    reconnections_at_dates = time_stats(events_list)
    time_spent = time_spent_at_date(probe=probe, start_date=start_date, end_date=end_date)
    keys_reconnections = reconnections_at_dates.keys()
    keys_dates = time_spent.keys()
    for key in keys_reconnections:
        if key in keys_dates:
            predicted = reconnections_at_dates['total number of reconnections'] * time_spent[key]
            if predicted * 0.7 < float(reconnections_at_dates[key]) < 1.3 * predicted:
                print('as predicted for ', key, 'with', reconnections_at_dates[key], 'instead of', predicted)
            else:
                print('not as predicted for ', key, 'with', reconnections_at_dates[key], 'instead of', predicted)


def analyse_by_radii(events_list: List[datetime], probe: int, start_date: str, end_date: str):
    reconnections_at_radii = distances_stats(events_list, probe)
    time_spent = time_spent_at_distances(probe, start_date, end_date)
    keys_reconnections = reconnections_at_radii.keys()
    keys_dists = time_spent.keys()
    for key in keys_reconnections:
        if key in keys_dists:
            predicted = reconnections_at_radii['total number of reconnections'] * time_spent[key]
            if predicted * 0.7 < float(reconnections_at_radii[key]) < 1.3 * predicted:
                print('as predicted for ', key,' with', reconnections_at_radii[key], 'instead of ', predicted)
            else:
                print('not as predicted for ', key, 'with', reconnections_at_radii[key], 'instead of ', predicted)


def get_events_dates(file_name: str):
    event_dates = []
    with open(file_name) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year, month, day = np.int(row['year']), np.int(row['month']), np.int(row['day'])
            hours, minutes, seconds = np.int(row['hours']), np.int(row['minutes']), np.int(row['seconds'])
            # if row['satisfied tests']:
            #     if row['satisfied tests'] == 'True':
            #         event_dates.append(datetime(year, month, day, hours, minutes, seconds))
            # else:
            #     event_dates.append(datetime(year, month, day, hours, minutes, seconds))
            event_dates.append(datetime(year, month, day, hours, minutes, seconds))
    return event_dates


# def amount_of_missing_data(probe: int, start_time: str, end_time: str, interval: float = 1, mode: str = 'date'):
#     duration = datetime.strptime(end_time, '%d/%m/%Y') - datetime.strptime(start_time, '%d/%m/%Y')
#     start_time = datetime.strptime(start_time, '%d/%m/%Y')
#     duration = np.int(duration.total_seconds() / 3600)
#     missing_data = {}
#     data_sets = {}
#     if mode == 'radius':
#         missing_data = {'<0.3': 0, '0.3 to 0.4': 0, '0.4 to 0.5': 0, '0.5 to 0.6': 0, '0.6 to 0.7': 0,
#                         '0.7 to 0.8': 0, '0.8 to 0.9': 0, '> 0.9': 0}
#         data_sets = {'<0.3': [], '0.3 to 0.4': [], '0.4 to 0.5': [], '0.5 to 0.6': [], '0.6 to 0.7': [],
#                      '0.7 to 0.8': [], '0.8 to 0.9': [], '> 0.9': []}
#     radius = 0
#     hour_interval = np.int(interval * 24)
#     for n in range(np.int(duration / hour_interval)):
#         try:
#             data = ImportedData(probe=probe, start_date=start_time.strftime('%d/%m/%Y'), duration=hour_interval)
#             if mode == 'date':
#                 if str(start_time.year) not in data_sets.keys():
#                     data_sets[str(start_time.year)] = []
#                     data_sets[str(start_time.year)].append(data)
#                 else:
#                     data_sets[str(start_time.year)].append(data)
#             if mode == 'radius':
#                 radius = data.data['r_sun'].values[0]
#                 if radius < 0.3:
#                     data_sets['<0.3 '].append(data)
#                 elif radius < 0.4:
#                     data_sets['0.3 to 0.4'].append(data)
#                 elif radius < 0.5:
#                     data_sets['0.4 to 0.5'].append(data)
#                 elif radius < 0.6:
#                     data_sets['0.5 to 0.6'].append(data)
#                 elif radius < 0.7:
#                     data_sets['0.6 to 0.7'].append(data)
#                 elif radius < 0.8:
#                     data_sets['0.7 to 0.8'].append(data)
#                 elif radius < 0.9:
#                     data_sets['0.8 to 0.9 au'].append(data)
#                 else:
#                     data_sets['above 0.9'].append(data)
#         except Exception:
#             print('exception')
#             if mode == 'radius':
#                 if radius < 0.3:
#                     missing_data['<0.3 '] += 1
#                 elif radius < 0.4:
#                     missing_data['0.3 to 0.4'] += 1
#                 elif radius < 0.5:
#                     missing_data['0.4 to 0.5'] += 1
#                 elif radius < 0.6:
#                     missing_data['0.5 to 0.6'] += 1
#                 elif radius < 0.7:
#                     missing_data['0.6 to 0.7'] += 1
#                 elif radius < 0.8:
#                     missing_data['0.7 to 0.8'] += 1
#                 elif radius < 0.9:
#                     missing_data['0.8 to 0.9 au'] += 1
#                 else:
#                     missing_data['above 0.9'] += 1
#
#         start_time = start_time + timedelta(hours=hour_interval)
#     if mode == 'date':
#         for key in data_sets.keys():
#             theory_length = 365 / interval
#             if theory_length == len(data_sets[key]):
#                 print('everything according to plan')
#             else:
#                 missing_data[key] = (theory_length - len(data_sets[key])) / theory_length
#     if mode == 'radius':
#         theory_length = 365 / interval
#         for key in data_sets.keys():
#             missing_data[key] = missing_data[key] / theory_length
#     return missing_data


def filecount(probe: int, year: int = 0):
    dir = r"C:\Users\tilquin\heliopy\data\helios\E1_experiment\New_proton_corefit_data_2017\ascii\helios" + str(probe)
    if probe == 1:
        if 1974 <= year <= 1984:
            dir = dir + '\\' + str(year)
    if probe == 2:
        if 1976 <= year <= 1979:
            dir = dir + '\\' + str(year)

    fls = [files for r, d, files in os.walk(dir) if files]
    file_number = sum([len(files) for r, d, files in os.walk(dir) if files])
    return file_number


if __name__ == '__main__':
    # need missing data list in order to determine exactly if events follow theory
    # HELIOS 1
    # {'1974': 0.9534246575342465, '1975': 0.1643835616438356, '1976': 0.2958904109589041,
    # '1977': 0.13424657534246576, '1978': 0.3561643835616438, '1979': 0.3835616438356164,
    # '1980': 0.14246575342465753, '1981': 0.5342465753424658, '1982': 0.8246575342465754,
    # '1983': 0.915068493150685, '1984': 0.8301369863013699}
    probe = 1
    day_accuracy = 0.5
    # analysis_start_date = '15/12/1974'
    analysis_start_date = '17/01/1976'
    # analysis_end_date = '15/08/1984'
    analysis_end_date = '17/01/1979'
    # file = 'helios2_magrec.csv'
    file = 'reconnections_tests_90_110_h1.csv'
    events = get_events_dates(file)



    analyse_dates(events, probe, analysis_start_date, analysis_end_date)
    analyse_by_radii(events, probe, analysis_start_date, analysis_end_date)
    # missing = amount_of_missing_data(1, '15/12/1974', '15/08/1984', day_accuracy)
    # print(missing)
