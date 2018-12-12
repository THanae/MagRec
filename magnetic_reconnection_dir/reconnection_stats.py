import os
import pprint
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import chi2
# import data_handler.utils.plotting_utils  # plotting_utils are useful for large legends

from data_handler.data_importer.helios_data import HeliosData
from data_handler.orbit_with_spice import kernel_loader, orbit_times_generator, orbit_generator, get_orbiter
from magnetic_reconnection_dir.csv_utils import create_events_list_from_csv_files

months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
          'november', 'december']
radii_divisions = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
radii_names = ['less than 0.3 au', '0.3 to 0.4 au', '0.4 to 0.5 au', '0.5 to 0.6 au', '0.6 to 0.7 au', '0.7 to 0.8 au',
               '0.8 to 0.9 au', 'above 0.9 au']

global helios_dir
helios_dir = r"C:\Users\Hanae\heliopy\data\helios\E1_experiment\New_proton_corefit_data_2017\ascii\helios"
potential_files = [files for r, d, files in os.walk(helios_dir + str(1) + '\\' + str(1974))]
if not potential_files:
    raise ValueError('Please enter the location of the helios files on your computer')


def distances_stats(events_list: List[datetime], probe: int, only_stats: bool = True) -> dict:
    """
    :param events_list: list of reconnection events
    :param probe: 1 or 2 for Helios 1 or 2
    :param only_stats: if True, only returns the number of events per distance, if False, also returns the dates
    :return: number of reconnection events at given distances from the sun
    """
    times_and_radii = {}
    for key in radii_names:
        times_and_radii[key] = []
    for event in events_list:
        start_time = event
        imported_data = HeliosData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                                   duration=1, probe=probe)
        try:
            radius = imported_data.data['r_sun'].loc[event]
        except ValueError:
            radius = np.mean(imported_data.data['r_sun'].values)
            print('exception in finding the radius')
        for n in range(len(radii_divisions)):
            radius_division = radii_divisions[len(radii_divisions) - n - 1]
            if radius > radius_division:
                radius_type = radii_names[len(radii_divisions) - n - 1]
                times_and_radii[radius_type].append([event, radius])
                break
    for key in times_and_radii.keys():
        if only_stats:
            times_and_radii[key] = len(times_and_radii[key])
        else:
            times_and_radii[key].append(str(len(times_and_radii[key])))
    times_and_radii['total number of reconnection events'] = len(events_list)
    pprint.pprint(times_and_radii)
    return times_and_radii


def time_stats(events_list: List[datetime], mode: str = 'yearly') -> dict:
    """
    :param events_list: list of reconnection events
    :param mode: yearly or monthly
    :return: dictionary of the number of reconnection events per year or per month
    """
    implemented_modes = ['yearly', 'monthly']
    times = {}
    if mode == 'yearly':
        for event in events_list:
            if str(event.year) not in times.keys():
                times[str(event.year)] = 1
            else:
                times[str(event.year)] += 1
    elif mode == 'monthly':
        for event in events_list:
            if str(event.year) not in times.keys():
                times[str(event.year)] = {}
                times[str(event.year)][months[event.month - 1]] = 1
            else:
                if months[event.month - 1] not in times[str(event.year)].keys():
                    times[str(event.year)][months[event.month - 1]] = 1
                else:
                    times[str(event.year)][months[event.month - 1]] += 1
    else:
        raise NotImplementedError('NO OTHER MODES IMPLEMENTED, CHOOSE FROM LIST', implemented_modes)
    times['total number of reconnection events'] = len(events_list)
    pprint.pprint(times)
    return times


def time_spent_at_distances(probe: int, start_date: str, end_date: str) -> dict:
    """
    :param probe: 1 or 2 for Helios 1 or 2
    :param start_date: start date of analysis
    :param end_date: end date of analysis
    :return: dictionary of the time spent per distance from the sun
    """
    orbiter = get_orbiter(start_time=start_date, end_time=end_date, probe=probe, interval=1)
    radii = np.array(np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2))
    time_spent = {}
    radius_types = radii_divisions + [100]
    for n in range(len(radii_divisions)):
        time_spent[radii_names[n]] = len(radii[np.all([radii >= radius_types[n], radii < radius_types[n + 1]], axis=0)])

    time_spent['total time'] = len(radii[np.all([radii < 1.2], axis=0)])
    for n in range(len(orbiter.times)):
        date = orbiter.times[n]
        _dir = helios_dir
        directory = _dir + str(probe) + '\\' + str(date.year)
        fls = [files for r, d, files in os.walk(directory) if files]
        day_of_year = date.strftime('%j')
        if 'h' + str(probe) + '_' + str(date.year) + '_' + str(day_of_year) + '_corefit.csv' not in fls[0]:
            radius = radii[n]
            time_spent['total time'] -= 1
            for m in range(len(radii_divisions)):
                radius_division = radii_divisions[len(radii_divisions) - m - 1]
                if radius > radius_division:
                    radius_type = radii_names[len(radii_divisions) - m - 1]
                    time_spent[radius_type] -= 1
                    break
            print('no such file: ', 'h' + str(probe) + '_' + str(date.year) + '_' + str(day_of_year) + '_corefit.csv')
    pprint.pprint(time_spent)
    return time_spent


def time_spent_at_date(probe: int, start_date: str, end_date: str, accuracy: float = 0.5, mode: str = 'yearly') -> dict:
    """
    :param probe: 1 or 2 for Helios 1 or 2
    :param start_date: start date of analysis
    :param end_date: end date of analysis
    :param accuracy: daily accuracy
    :param mode: yearly or monthly
    :return: time spent per year or per month
    """
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
            time_spent[key] = filecount(probe, int(key))[0] / filecount(probe)[0]
    elif mode == 'monthly':  # not very useful when there are few events but might be more useful with other spacecrafts
        for key in time_spent.keys():
            length, doy = filecount(probe, int(key))
            month_dict = get_month_dict(doy, int(key))
            time_spent[key] = month_dict
            print(month_dict)
            print('count', filecount(probe)[0])
            for keys in month_dict.keys():
                month_dict[keys] = month_dict[keys] / filecount(probe)[0]
    else:
        raise NotImplementedError('THIS MODE HAS NOT BEEN IMPLEMENTED, CHOOSE FROM ', implemented_modes)
    time_spent['total time'] = len(times) * accuracy  # in days
    pprint.pprint(time_spent)
    return time_spent


def analyse_dates(events_list: List[datetime], probe: int, start_date: str, end_date: str, mode: str = 'yearly'):
    """
    :param events_list: list of reconnection events
    :param probe: 1 or 2 for Helios 1 or 2
    :param start_date: start date of analysis
    :param end_date: end date of analysis
    :param mode: yearly or monthly
    :return:
    """
    events_at_dates = time_stats(events_list, mode=mode)
    time_spent = time_spent_at_date(probe=probe, start_date=start_date, end_date=end_date, mode=mode)
    keys_reconnection_events = events_at_dates.keys()
    keys_dates = time_spent.keys()
    if mode == 'yearly':
        for key in keys_reconnection_events:
            if key in keys_dates:
                predicted = events_at_dates['total number of reconnection events'] * time_spent[key]
                if predicted * 0.7 < float(events_at_dates[key]) < 1.3 * predicted:
                    print('as predicted for ', key, 'with', events_at_dates[key], 'instead of', predicted)
                else:
                    print('not as predicted for ', key, 'with', events_at_dates[key], 'instead of',
                          predicted)
    elif mode == 'monthly':
        for key in keys_reconnection_events:
            if key in keys_dates:
                for key_m in events_at_dates[key].keys():
                    if key_m in time_spent[key].keys():
                        predicted = events_at_dates['total number of reconnection events'] * time_spent[key][key_m]
                        if predicted * 0.7 < float(events_at_dates[key][key_m]) < 1.3 * predicted:
                            print('as predicted for ', key, key_m, 'with', events_at_dates[key][key_m],
                                  'instead of', predicted)
                        else:
                            print('not as predicted for ', key, key_m, 'with', events_at_dates[key][key_m],
                                  'instead of', predicted)


def analyse_by_radii(events_list: List[datetime], probe: int, start_date: str, end_date: str):
    """

    :param events_list: list of reconnection events
   :param probe: 1 or 2 for Helios 1 or 2
    :param start_date: start date of analysis
    :param end_date: end date of analysis
    :return:
    """
    reconnection_events_at_radii = distances_stats(events_list, probe)
    time_spent = time_spent_at_distances(probe, start_date, end_date)
    keys_reconnection_events = reconnection_events_at_radii.keys()
    keys_dists = time_spent.keys()
    for key in keys_reconnection_events:
        if key in keys_dists:
            predicted = reconnection_events_at_radii['total number of reconnection events'] * time_spent[key]
            if predicted * 0.7 < float(reconnection_events_at_radii[key]) < 1.3 * predicted:
                print('as predicted for ', key, ' with', reconnection_events_at_radii[key], 'instead of ', predicted)
            else:
                print('not as predicted for ', key, 'with', reconnection_events_at_radii[key], 'instead of ', predicted)


def get_month_dict(days_of_year: list, year: int) -> dict:
    """
    :param days_of_year: days of the year that we are analysing
    :param year: year to be analysed
    :return:
    """
    days_per_month = {'january': 0, 'february': 0, 'march': 0, 'april': 0, 'may': 0, 'june': 0, 'july': 0, 'august': 0,
                      'september': 0, 'october': 0, 'november': 0, 'december': 0}
    for doy in days_of_year:
        date = datetime(year, 1, 1) + timedelta(doy - 1)
        month = months[date.month - 1]
        days_per_month[month] += 1
    return days_per_month


def filecount(probe: int, year: int = 0):
    """
    :param probe: 1 or 2 for Helios 1 or 2
    :param year: year ot analyse, if within the probe mission dates, counts only the files in that year
    :return:
    """
    directory = helios_dir + str(probe)
    if probe == 1:
        if 1974 <= year <= 1984:
            directory = directory + '\\' + str(year)
    if probe == 2:
        if 1976 <= year <= 1979:
            directory = directory + '\\' + str(year)

    fls = [files for r, d, files in os.walk(directory) if files]
    days_of_year = []
    for file in fls[0]:
        doy = int(file[8] + file[9] + file[10])
        days_of_year.append(doy)
    file_number = sum([len(files) for r, d, files in os.walk(directory) if files])
    return file_number, days_of_year


def analyse_all_probes(mode: str = 'radius'):
    """
    Analyses the Helios 1 and Helios 2 data
    :param mode: yearly, monthly or radius
    :return:
    """
    events1 = create_events_list_from_csv_files([['helios1_magrec2.csv', None], ['helios1mag_rec3.csv', None]])
    events1 = [event for event in events1 if event.year < 1981 or (event.year + event.month) < 1990]  # before oct 1981
    events2 = create_events_list_from_csv_files([['helios2_magrec2.csv', None], ['helios2mag_rec3.csv', None]])
    if mode == 'radius':
        dis1, dis2 = distances_stats(events1, probe=1), distances_stats(events2, probe=2)
        for key in dis1.keys():
            if key in dis2.keys():
                dis1[key] = int(dis1[key]) + int(dis2[key])
        print(dis1)
        time_analysed1 = time_spent_at_distances(probe=1, start_date='15/12/1974', end_date='15/09/1981')
        # time_analysed1 = time_spent_at_distances(probe=1, start_date='15/12/1974', end_date='15/08/1984')
        time_analysed2 = time_spent_at_distances(probe=2, start_date='17/01/1976', end_date='17/01/1979')
        for key in time_analysed1.keys():
            if key in time_analysed2.keys():
                time_analysed1[key] = float(time_analysed1[key]) + float(time_analysed2[key])
        print(time_analysed1)

        reconnection_per_radius = {}

        def confidence_interval_poisson(k, alpha=0.36):
            down = 0.5 * chi2.ppf(alpha / 2, 2 * k) if k != 0 else 0
            up = 0.5 * chi2.ppf(1 - alpha / 2, 2 * k + 2)
            return down, up
        poisson_errors = []
        for key in time_analysed1.keys():
            if key in dis1.keys():
                _lambda = float(dis1[key]) / float(time_analysed1[key])
                time_t = float(time_analysed1[key])
                reconnection_per_radius[key] = _lambda
                lower, upper = confidence_interval_poisson(dis1[key])
                poisson_errors.append(tuple([_lambda - lower/time_t, upper/time_t - _lambda]))
        poisson_errors = [_ for _ in zip(*poisson_errors)]
        pprint.pprint(reconnection_per_radius)
        fig1 = plt.figure()
        plot_trend(reconnection_per_radius, errors=poisson_errors)
        plt.title('Normalised rate of reconnection events ')
        plt.ylabel('Rate per day')
        fig2 = plt.figure()
        plot_trend(dis1)
        fig3 = plt.figure()
        plot_trend(time_analysed1)
        plt.show()
    elif mode == 'time':
        # not very sensible to use it as Helios 2 was working only part of the time when Helios 1 was working
        time1 = time_stats(events1)
        time2 = time_stats(events2)
        for key in time1.keys():
            if key in time2.keys():
                time1[key] = int(time1[key]) + int(time2[key])
        print(time1)
        plot_trend(time1)


def plot_trend(stat: dict, mode='yearly', errors: Optional[List] = None):
    """
    Plots histograms of the obtained data sets
    :param stat: dictionary of stats that will be plotted
    :param mode: yearly, monthly or radius
    :return:
    """
    implemented_modes = ['yearly', 'monthly', 'radius']
    if mode == 'yearly' or mode == 'radius':
        # stat.pop('total number of reconnection events')
        if errors is None:
            plt.bar(range(len(stat)), list(stat.values()), align='center')
        else:
            plt.bar(range(len(stat)), list(stat.values()), align='center', yerr=errors)
        plt.xticks(range(len(stat)), list(stat.keys()), rotation=20)
    elif mode == 'monthly':
        new_stat = {}
        stat.pop('total number of reconnection events', None)
        for key in stat.keys():
            for key_m in stat[key].keys():
                new_stat[key_m[0] + key_m[1] + key_m[2] + '_' + key[2] + key[3]] = stat[key][key_m]

        plt.bar(range(len(new_stat)), new_stat.values(), align='center')
        plt.xticks(range(len(new_stat)), list(new_stat.keys()))
    else:
        print('THIS MODE IS NOT IMPLEMENTED, CHOOSE FROM ', implemented_modes)
    # plt.show()


if __name__ == '__main__':
    # mode = 'monthly'

    # analysis = [{'probe': 1, 'start_date': '15/12/1974', 'end_date': '15/08/1984',
    #              'events': create_events_list_from_csv_files([['helios1_magrec2.csv', None], ['helios1mag_rec3.csv',
    #                                                                                           None]])},
    #             {'probe': 2, 'start_date': '17/01/1976', 'end_date': '17/01/1979',
    #              'events': create_events_list_from_csv_files([['helios2_magrec2.csv', None], ['helios2mag_rec3.csv',
    #                                                                                           None]])},
    #             {'probe': 'ulysses', 'start_date': '01/01/1992', 'end_date': '12/12/2009',
    #              'events': create_events_list_from_csv_files([['mag_rec_ulysses.csv', None]])}
    #             ]

    # spacecraft_to_analyse = analysis[2]
    # space_probe, events = spacecraft_to_analyse['probe'], spacecraft_to_analyse['events']
    # analysis_start_date, analysis_end_date = spacecraft_to_analyse['start_date'], spacecraft_to_analyse['end_date']
    # stats = time_stats(events, mode='yearly')
    # plot_trend(stats, mode='yearly')

    analyse_all_probes(mode='radius')

    # st = time_spent_at_date(start_date=analysis_start_date, end_date=analysis_end_date, probe=space_probe)
    # print(st.pop('total time', None))
    # plot_trend(st, mode)
