import os
from datetime import timedelta, datetime
from typing import List

from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from dateutil.relativedelta import relativedelta

from data_handler.data_importer.helios_data import HeliosData
from data_handler.orbit_with_spice import get_planet_orbit, get_orbiter
from magnetic_reconnection_dir.csv_utils import get_dates_from_csv
from magnetic_reconnection_dir.reconnection_stats import time_stats


def plot_hist_dist(orbiter, spacecraft, stat, planet=None, events=None, missing_data=None, plot_sun: bool = False):
    quantity_support()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if plot_sun:
        sun_distance = np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2)
        ax.plot(orbiter.times, sun_distance, label='Helios' + str(spacecraft) + ' orbit')
        if events is not None:
            plot_event_info(ax, sun_distance, events, spacecraft)

    ax.set_title('Helios ' + str(spacecraft) + '  between ' + str(orbiter.times[0]) + ' and ' + str(orbiter.times[-1]))

    if planet is not None:
        orbiter_planet = get_planet_orbit(planet, datetime.strftime(orbiter.times[0], '%d/%m/%Y'),
                                          datetime.strftime(orbiter.times[-1] + timedelta(days=1), '%d/%m/%Y'))
        spacecraft_planet = np.sqrt(
            (orbiter.x - orbiter_planet.x) ** 2 + (orbiter.y - orbiter_planet.y) ** 2 + (
                        orbiter.z - orbiter_planet.z) ** 2)
        print(len(orbiter.x), len(orbiter_planet.x))
        ax.set_ylabel('Distance between spacecraft and ' + planet)
        ax.plot(orbiter_planet.times, spacecraft_planet, label=planet + '-Spacecraft distance')
        if events is not None:
            plot_event_info(ax, spacecraft_planet, events, spacecraft)
    spacecraft_legend = ax.legend(loc=4)
    plt.gca().add_artist(spacecraft_legend)

    if missing_data is not None:
        for year, month in missing_data:
            _missing = datetime(year, month, 15)
            ax.axvline(x=_missing, linewidth=10, color='m', alpha=0.4)

    black_cross = mlines.Line2D([], [], color='k', marker='+', linestyle='None', label='Reconnection event density')
    black_dot = mlines.Line2D([], [], color='k', marker='o', linestyle='None', label='Reconnection event speed')
    patches = []
    types = ['very low', 'low', 'medium', 'high', 'very high']
    for loop in range(5):
        patches.append(mpatches.Patch(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][loop], label=types[loop]))

    ax.legend(handles=[black_cross, black_dot] + patches, loc=1)
    ax = ax.twinx()

    new_stat = {}
    stat.pop('total number of reconnections', None)
    maximum_reconnection = 0
    for key in stat.keys():
        for key_m in stat[key].keys():
            new_stat[key_m[0] + key_m[1] + key_m[2] + '_' + key[2] + key[3]] = stat[key][key_m]
            if stat[key][key_m] > maximum_reconnection:
                maximum_reconnection = stat[key][key_m]
    print('Maximum number of reconnection events during a month', maximum_reconnection)
    ax.set_ylabel('Number of reconnections /' +str(maximum_reconnection))
    datetime_keys = []
    for key in new_stat.keys():
        k = str_to_datetime(key)
        datetime_keys.append(k)
        ax.axvline(x=k, linewidth=10, color='k', ymax=new_stat[key]/maximum_reconnection, alpha=0.4)

    plt.show()


def plot_event_info(ax, spacecraft_planet, events: List[datetime], spacecraft):
    args = {}
    normal_size = 8
    for event in events:
        try:
            density, speed = find_density_and_speed(event, spacecraft)
            classification = classify_wind(density, speed)
            if not np.isnan(density):
                dens_color = find_color_from_type(classification[0])
                speed_color = find_color_from_type(classification[1])
            else:
                dens_color = 'k'
                speed_color = 'k'
        except RuntimeWarning:
            dens_color = 'k'
            speed_color = 'k'
        arg = orbiter.times.index(datetime(event.year, event.month, event.day))
        ax.plot(orbiter.times[arg], spacecraft_planet[arg], marker='o', color=speed_color, markersize=normal_size,
                alpha=0.7)
        ax.plot(orbiter.times[arg], spacecraft_planet[arg], marker='+', color=dens_color, markersize=normal_size, mew=5)

        if str(arg) in list(args.keys()):
            args[str(arg)] += 1
            ax.plot(orbiter.times[arg], spacecraft_planet[arg], marker='o', color=speed_color,
                    markersize=normal_size * args[str(arg)], alpha=0.7)
            ax.plot(orbiter.times[arg], spacecraft_planet[arg], marker='+', color=dens_color,
                    markersize=normal_size * args[str(arg)], mew=5)
        else:
            args[str(arg)] = 1


def str_to_datetime(date: str):
    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
             'nov': 11, 'dec': 12}
    month = months[date[0]+date[1]+date[2]]
    year = str(19) + date[4] + date[5]
    _date = datetime(int(year), int(month), 15)
    return _date


def get_events_dates(probe: int):
    if probe == 1:
        events = get_dates_from_csv('helios1_magrec2.csv')
        for event in get_dates_from_csv('helios1mag_rec3.csv'):
            if event not in events:
                events.append(event)
    elif probe == 2:
        events = get_dates_from_csv('helios2_magrec2.csv')
        for event in get_dates_from_csv('helios2mag_rec3.csv'):
            if event not in events:
                events.append(event)
    else:
        events = []
    return events


def classify_wind(density, speed):
    if speed < 200:
        speed_type = 'very low'
    elif speed < 300:
        speed_type = 'low'
    elif speed < 400:
        speed_type = 'medium'
    elif speed < 500:
        speed_type = 'high'
    else:
        speed_type = 'very high'
    if density < 2:
        density_type = 'vey low'
    elif density < 10:
        density_type = 'low'
    elif density < 100:
        density_type = 'medium'
    elif density < 200:
        density_type = 'high'
    else:
        density_type = 'very high'
    return density_type, speed_type


def find_color_from_type(solar_wind_characteristic: str):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # colors = plt.cm.Set3(np.linspace(0, 1, 12))
    if solar_wind_characteristic == 'very low':
        color = colors[0]
    elif solar_wind_characteristic == 'low':
        color = colors[1]
    elif solar_wind_characteristic == 'medium':
        color = colors[2]
    elif solar_wind_characteristic == 'high':
        color = colors[3]
    else:
        color = colors[4]
    return color


def find_density_and_speed(event, probe):
    start_analysis = event - timedelta(minutes=60)
    imported_data = HeliosData(start_date=start_analysis.strftime('%d/%m/%Y'), duration=2, probe=probe)
    imported_data.data.dropna(inplace=True)
    density = np.mean(imported_data.data['n_p'].values)
    imported_data.create_processed_column('vp_magnitude')
    speed = np.mean(imported_data.data['vp_magnitude'].values)
    print(speed, density)
    if np.isnan(speed) or np.isnan(density):
        print(event)
        print(imported_data.data['n_p'])
    return density, speed


def find_data_gaps(probe, start_time:str, end_time:str):
    start = datetime.strptime(start_time, '%d/%m/%Y')
    start = datetime(start.year, start.month, 1)
    end = datetime.strptime(end_time, '%d/%m/%Y')
    interval = relativedelta(months=1)
    directory = r"C:\Users\tilquin\heliopy\data\helios\E1_experiment\New_proton_corefit_data_2017\ascii\helios" + str(
        probe)
    date = start
    missing_data = []
    while date < end:
        year, month, day = date.year, date.month, date.day
        _dir = directory + '\\' + str(year)
        fls = [files for r, d, files in os.walk(_dir) if files]

        date1 = date
        date2 = date + interval - timedelta(days=1)
        doy = []
        for days in range(int((date2 - date1).total_seconds()/(3600 * 24))):
            doy.append(datetime.strftime(date1, '%j'))
            date1 = date1 + timedelta(days=1)
        missing = 0
        for day_of_year in doy:
            if 'h' + str(probe) + '_' + str(year) + '_' + str(day_of_year) + '_corefit.csv' in (np.array(fls).flatten()):
                missing = missing
            else:
                missing += 1
        if missing > 25:
            missing_data.append([int(year), int(month)])
        date = date + interval
    print(missing_data)
    return missing_data


if __name__ == '__main__':
    # probe = 1
    # start_time = '15/12/1974'
    # end_time = '15/08/1984'
    probe = 2
    start_time = '20/01/1976'
    end_time = '01/10/1979'
    planet = 'Earth'
    dates = get_events_dates(probe)
    orbiter = get_orbiter(probe, start_time, end_time)
    missing = find_data_gaps(probe, start_time, end_time)
    stats = time_stats(dates, mode='monthly')
    plot_hist_dist(orbiter, probe, stats, planet, dates, missing)
