import csv
import os
import pprint
import numpy as np
from typing import List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_handler.data_importer.helios_data import HeliosData
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

        imported_data = HeliosData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                                   duration=1, probe=probe)
        try:
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
        if only_stats:
            times_and_radii[key] = (str(len(times_and_radii[key])))
        else:
            times_and_radii[key].append(str(len(times_and_radii[key])))
    times_and_radii['total number of reconnections'] = len(events_list)
    pprint.pprint(times_and_radii)
    return times_and_radii


def time_stats(events_list: List[datetime], stats_mode: bool = True, mode: str = 'yearly') -> dict:
    implemented_modes = ['yearly', 'monthly']
    times = {}
    if mode == 'yearly':
        for event in events_list:
            if str(event.year) not in times.keys():
                times[str(event.year)] = []
                times[str(event.year)].append(event)
            else:
                times[str(event.year)].append(event)
        for key in times.keys():
            if stats_mode:
                times[key] = str(len(times[key]))
            else:
                times[key].append(str(len(times[key])))

        times['total number of reconnections'] = len(events_list)
    elif mode == 'monthly':
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                  'november', 'december']
        for event in events_list:
            if str(event.year) not in times.keys():
                times[str(event.year)] = {}
                times[str(event.year)][months[event.month - 1]] = 0
                times[str(event.year)][months[event.month - 1]] += 1
            else:
                if months[event.month - 1] not in times[str(event.year)].keys():
                    times[str(event.year)][months[event.month - 1]] = 0
                    times[str(event.year)][months[event.month - 1]] += 1
                else:
                    times[str(event.year)][months[event.month - 1]] += 1
        times['total number of reconnections'] = len(events_list)
    else:
        print('NO OTHER MODES IMPLEMENTED, CHOOSE FROM LIST', implemented_modes)

    pprint.pprint(times)
    return times


def time_spent_at_distances(probe: int, start_date: str, end_date: str, accuracy: float = 0.5) -> dict:
    orbiter = kernel_loader(probe)
    times = orbit_times_generator(start_date, end_date, interval=accuracy)
    orbit_generator(orbiter, times)
    radii = np.array(np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2))
    time_spent = {}
    time_analysed = datetime.strptime(end_date, '%d/%m/%Y') - datetime.strptime(start_date, '%d/%m/%Y')
    time_analysed = int(time_analysed.total_seconds() / (3600 * 24))
    time_spent['total time'] = len(radii[np.all([radii < 1.2], axis=0)])  # * filecount(probe)/time_analysed
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


def time_spent_at_date(probe: int, start_date: str, end_date: str, accuracy: float = 0.5, mode: str = 'yearly') -> dict:
    implemented_modes = ['yearly', 'monthly']
    orbiter = kernel_loader(probe)
    times = orbit_times_generator(start_date, end_date, interval=accuracy)
    orbit_generator(orbiter, times)
    time_spent = {}
    for time in times:
        if str(time.year) not in time_spent.keys():
            time_spent[str(time.year)] = 1
        else:
            time_spent[str(time.year)] += 1

    if mode == 'yearly':
        for key in time_spent.keys():
            # time_spent[key] = (time_spent[key] / len(times)) * filecount(probe, int(key))/365
            time_spent[key] = filecount(probe, int(key))[0] / filecount(probe)[0]
    elif mode == 'monthly':  # not very useful when there are so few reconnections to start with,
        #  but might be more useful with other spacecrafts
        for key in time_spent.keys():
            length, doy = filecount(probe, int(key))
            month_dict = get_month_dict(doy, int(key))
            time_spent[key] = month_dict
            print(month_dict)
            print('count', filecount(probe)[0])
            for keys in month_dict.keys():
                month_dict[keys] = month_dict[keys] / filecount(probe)[0]

    else:
        print('THIS MODE HAS NOT BEEN IMPLEMENTED, CHOOSE FROM ', implemented_modes)
    time_spent['total time'] = len(times) * accuracy  # in days

    pprint.pprint(time_spent)
    return time_spent


def analyse_dates(events_list: List[datetime], probe: int, start_date: str, end_date: str, mode: str = 'yearly'):
    reconnections_at_dates = time_stats(events_list, mode=mode)
    time_spent = time_spent_at_date(probe=probe, start_date=start_date, end_date=end_date, mode=mode)
    keys_reconnections = reconnections_at_dates.keys()
    keys_dates = time_spent.keys()
    if mode == 'yearly':
        for key in keys_reconnections:
            if key in keys_dates:
                predicted = reconnections_at_dates['total number of reconnections'] * time_spent[key]
                if predicted * 0.7 < float(reconnections_at_dates[key]) < 1.3 * predicted:
                    print('as predicted for ', key, 'with', reconnections_at_dates[key], 'instead of', predicted)
                else:
                    print('not as predicted for ', key, 'with', reconnections_at_dates[key], 'instead of',
                          predicted)
    elif mode == 'monthly':
        for key in keys_reconnections:

            if key in keys_dates:
                for key_m in reconnections_at_dates[key].keys():
                    if key_m in time_spent[key].keys():
                        predicted = reconnections_at_dates['total number of reconnections'] * time_spent[key][key_m]
                        if predicted * 0.7 < float(reconnections_at_dates[key][key_m]) < 1.3 * predicted:
                            print('as predicted for ', key, key_m, 'with', reconnections_at_dates[key][key_m],
                                  'instead of', predicted)
                        else:
                            print('not as predicted for ', key, key_m, 'with', reconnections_at_dates[key][key_m],
                                  'instead of',
                                  predicted)


def analyse_by_radii(events_list: List[datetime], probe: int, start_date: str, end_date: str):
    reconnections_at_radii = distances_stats(events_list, probe)
    time_spent = time_spent_at_distances(probe, start_date, end_date)
    keys_reconnections = reconnections_at_radii.keys()
    keys_dists = time_spent.keys()
    for key in keys_reconnections:
        if key in keys_dists:
            predicted = reconnections_at_radii['total number of reconnections'] * time_spent[key]
            if predicted * 0.7 < float(reconnections_at_radii[key]) < 1.3 * predicted:
                print('as predicted for ', key, ' with', reconnections_at_radii[key], 'instead of ', predicted)
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


def get_month_dict(days_of_year: list, year: int) ->dict:
    days_per_month = {'january': 0, 'february': 0, 'march': 0, 'april': 0, 'may': 0, 'june': 0, 'july': 0, 'august': 0,
                      'september': 0, 'october': 0, 'november': 0, 'december': 0}
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
    dates = []
    for doy in days_of_year:
        date = datetime(year, 1, 1) + timedelta(doy - 1)
        dates.append(date)
    for date in dates:
        month = months[date.month - 1]
        days_per_month[month] += 1
    return days_per_month


def filecount(probe: int, year: int = 0):
    dir = r"C:\Users\tilquin\heliopy\data\helios\E1_experiment\New_proton_corefit_data_2017\ascii\helios" + str(probe)
    if probe == 1:
        if 1974 <= year <= 1984:
            dir = dir + '\\' + str(year)
    if probe == 2:
        if 1976 <= year <= 1979:
            dir = dir + '\\' + str(year)

    fls = [files for r, d, files in os.walk(dir) if files]
    days_of_year = []
    for file in fls[0]:
        doy = int(file[8] + file[9] + file[10])
        days_of_year.append(doy)
    file_number = sum([len(files) for r, d, files in os.walk(dir) if files])
    return file_number, days_of_year


def get_radius(events_list: List[datetime], year: int = 1976, month: int = 0, probe: int = 2) ->list:
    """
    Finds the position of events compared to the sun duing a given year (and given month)
    :param events_list: list of events to be analysed
    :param year: year to be analysed
    :param month: month to be considered (all year if month = 0)
    :param probe: 1 or 2 for Helios 1 or 2
    :return:
    """
    time_radius = []
    for event in events_list:
        if event.year == year:
            if month == 0 or event.month == month:
                start_time = event
                imported_data = HeliosData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                                             duration=1, probe=probe)
                radius = imported_data.data['r_sun'].loc[event]
                time_radius.append([event, radius])
    print(time_radius)
    return time_radius


def plot_trend(stat: dict, mode='yearly'):
    """
    Plots histograms of the obtained data sets
    :param stat: dictionary of stats that will be plotted
    :param mode: 'yearly' for a yearly analysis dict, 'monthly' for a monthly analysis dict, and 'radius' for a distance
                analysis dict
    :return:
    """
    implemented_modes = ['yearly', 'monthly', 'radius']
    if mode == 'yearly' or mode == 'radius':
        plt.bar(range(len(stat)), stat.values(), align='center')
        plt.xticks(range(len(stat)), list(stat.keys()))
    elif mode == 'monthly':
        new_stat = {}
        stat.pop('total number of reconnections', None)
        for key in stat.keys():
            for key_m in stat[key].keys():
                new_stat[key_m[0] + key_m[1] + key_m[2] + '_' + key[2] + key[3]] = stat[key][key_m]

        plt.bar(range(len(new_stat)), new_stat.values(), align='center')
        plt.xticks(range(len(new_stat)), list(new_stat.keys()))
    else:
        print('THIS MODE IS NOT IMPLEMNETED, CHOOSE FROM ', implemented_modes)
    plt.show()


if __name__ == '__main__':
    # need missing data list in order to determine exactly if events follow theory
    # HELIOS 1
    # {'1974': 0.9534246575342465, '1975': 0.1643835616438356, '1976': 0.2958904109589041,
    # '1977': 0.13424657534246576, '1978': 0.3561643835616438, '1979': 0.3835616438356164,
    # '1980': 0.14246575342465753, '1981': 0.5342465753424658, '1982': 0.8246575342465754,
    # '1983': 0.915068493150685, '1984': 0.8301369863013699}
    # maybe could combine data from the two probes in order to have more data...
    # but then if one is skewed the other one is skewed too
    probe = 1
    day_accuracy = 0.5
    analysis_start_date = '15/12/1974'
    # analysis_start_date = '17/01/1976'
    analysis_end_date = '15/08/1984'
    # analysis_end_date = '17/01/1979'
    file = 'helios1_magrec.csv'
    mode = 'yearly'
    # file = 'helios2_magrec.csv'
    events = get_events_dates(file)
    # st = time_spent_at_date(start_date=analysis_start_date, end_date=analysis_end_date, probe=probe)
    # print(st.pop('total time', None))
    # plot_trend(st, mode)
    # time_spent_at_date(probe, start_date=analysis_start_date, end_date=analysis_end_date, mode=mode)
    analyse_by_radii(events, probe, analysis_start_date, analysis_end_date)
    analyse_dates(events, probe, analysis_start_date, analysis_end_date, mode='monthly')
