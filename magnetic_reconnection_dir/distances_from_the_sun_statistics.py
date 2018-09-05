import os
from datetime import timedelta, datetime
from typing import List, Optional
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as m_lines
import matplotlib.patches as m_patches
from dateutil.relativedelta import relativedelta

from data_handler.data_importer.helios_data import HeliosData
from data_handler.orbit_with_spice import get_planet_orbit, get_orbiter
from data_handler.utils.column_processing import get_outliers, get_derivative
from magnetic_reconnection_dir.csv_utils import get_dates_from_csv
from magnetic_reconnection_dir.reconnection_stats import time_stats

proton_mass = 1.6e-27
mu_0 = np.pi * 4e-7
solar_types = ['very low', 'low', 'medium', 'high', 'very high']


def plot_hist_dist(orbiter, spacecraft, stat, planet: Optional[str] = None, events: Optional[List[datetime]] = None,
                   missing_data: Optional[list] = None, plot_sun: bool = False):
    quantity_support()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if plot_sun:
        sun_distance = np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2)
        ax.plot(orbiter.times, sun_distance, label='Helios' + str(spacecraft) + ' orbit')
        if events is not None:
            plot_event_info(ax, orbiter, sun_distance, events, spacecraft)

    ax.set_title('Helios ' + str(spacecraft) + '  between ' + str(orbiter.times[0]) + ' and ' + str(orbiter.times[-1]))

    if planet is not None:
        orbiter_planet = get_planet_orbit(planet, datetime.strftime(orbiter.times[0], '%d/%m/%Y'),
                                          datetime.strftime(orbiter.times[-1] + timedelta(days=1), '%d/%m/%Y'))
        spacecraft_planet = np.sqrt(
            (orbiter.x - orbiter_planet.x) ** 2 + (orbiter.y - orbiter_planet.y) ** 2 + (
                    orbiter.z - orbiter_planet.z) ** 2)
        ax.set_ylabel('Distance between spacecraft and ' + planet)
        ax.plot(orbiter_planet.times, spacecraft_planet, label=planet + '-Spacecraft distance')
        if events is not None:
            plot_event_info(ax, orbiter, spacecraft_planet, events, spacecraft)
    spacecraft_legend = ax.legend(loc=4)
    plt.gca().add_artist(spacecraft_legend)

    if missing_data is not None:
        for year, month in missing_data:
            _missing = datetime(year, month, 15)
            ax.axvline(x=_missing, linewidth=10, color='blue', alpha=0.2)

    black_cross = m_lines.Line2D([], [], color='k', marker='+', linestyle='None', label='Reconnection event density')
    black_dot = m_lines.Line2D([], [], color='k', marker='o', linestyle='None', label='Reconnection event speed')
    patches = []
    for m in range(5):
        patches.append(m_patches.Patch(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][m],
                                       label=solar_types[m]))

    ax.legend(handles=[black_cross, black_dot] + patches, loc=1)
    ax = ax.twinx()

    new_stat = {}
    stat.pop('total number of reconnection events', None)
    maximum_reconnection = 0
    for key in stat.keys():
        for key_m in stat[key].keys():
            new_stat[key_m[0] + key_m[1] + key_m[2] + '_' + key[2] + key[3]] = stat[key][key_m]
            if stat[key][key_m] > maximum_reconnection:
                maximum_reconnection = stat[key][key_m]
    print('Maximum number of reconnection events during a month', maximum_reconnection)
    ax.set_ylabel('Number of reconnection events /' + str(maximum_reconnection))
    datetime_keys = []
    for key in new_stat.keys():
        k = str_to_datetime(key)
        datetime_keys.append(k)
        ax.axvline(x=k, linewidth=10, color='k', ymax=new_stat[key] / maximum_reconnection, alpha=0.4)
    plt.show()


def plot_event_info(ax, orbiter, spacecraft_planet: np.ndarray, events: List[datetime], spacecraft: int,
                    plot_each_point: bool = False):
    args = {}
    normal_size = 10
    for event in events:
        density, speed = find_density_and_speed(event, spacecraft)
        classification = classify_wind(density, speed)
        if not np.isnan(density):
            dens_color = find_color_from_type(classification[0])
            speed_color = find_color_from_type(classification[1])
        else:
            dens_color = 'k'
            speed_color = 'k'

        arg = orbiter.times.index(datetime(event.year, event.month, event.day))
        if plot_each_point:
            ax.plot(event, spacecraft_planet[arg], marker='o', color=speed_color, markersize=normal_size, alpha=0.8)
            ax.plot(event, spacecraft_planet[arg], marker='+', color=dens_color, markersize=normal_size, mew=3)
        else:
            ax.plot(orbiter.times[arg], spacecraft_planet[arg], marker='o', color=speed_color, markersize=normal_size,
                    alpha=0.8)
            ax.plot(orbiter.times[arg], spacecraft_planet[arg], marker='+', color=dens_color, markersize=normal_size,
                    mew=3)
            if str(arg) in list(args.keys()):
                args[str(arg)] += 1
                ax.plot(orbiter.times[arg], spacecraft_planet[arg], marker='o', color=speed_color,
                        markersize=normal_size * args[str(arg)], alpha=0.8)
                ax.plot(orbiter.times[arg], spacecraft_planet[arg], marker='+', color=dens_color,
                        markersize=normal_size * args[str(arg)], mew=4)
            else:
                args[str(arg)] = 1


def str_to_datetime(date: str) ->datetime:
    """
    Transforms a date string to a datetime object
    :param date: date string
    :return: the date as a datetime object
    """
    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
              'nov': 11, 'dec': 12}
    month = months[date[0] + date[1] + date[2]]
    year = str(19) + date[4] + date[5]
    _date = datetime(int(year), int(month), 15)
    return _date


def get_events_dates(probe: int) ->List[datetime]:
    """
    Gets all events dates for a given probe
    :param probe: 1 or 2 for Helios 1 or 2
    :return: list of events dates
    """
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


def classify_wind(density: float, speed: float) ->List[str, str]:
    """
    Classifies characteristics of the solar wind
    :param density: proton density
    :param speed: speed of the solar wind
    :return: list of the density and speed types
    """
    speed_types = [0, 300, 350, 400, 500]
    density_types = [0, 6, 30, 60, 150]
    speed_type, density_type = 'place_holder', 'place_holder'
    for n in range(len(speed_types)):
        if speed > speed_types[n]:
            speed_type = solar_types[n]

    for n in range(len(density_types)):
        if density > density_types[n]:
            density_type = solar_types[n]

    return [density_type, speed_type]


def find_color_from_type(solar_wind_characteristic: str) ->str:
    """
    Links a characteristic to a color
    :param solar_wind_characteristic: characteristic to associate a color to
    :return: the color associated with the given type
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_type_number = solar_types.index(solar_wind_characteristic)
    color = colors[color_type_number]
    return color


def histogram_speed_density():
    n, v, b, rad = [], [], [], []
    for loop in range(2):
        _probe = loop + 1
        _events = get_events_dates(_probe)
        for _event in _events:
            density, speed, radius, t_perp, t_par, t_tot, b_field = find_density_and_speed(_event, _probe,
                                                                                           distance=True)
            n.append(density)
            v.append(speed)
            b.append(b_field)
            rad.append(radius)

    plt.hist(n, bins=25)
    plt.show()
    plt.hist(v, bins=25, width=5)
    plt.show()
    plt.hist(b, bins=25)
    plt.show()
    plt.hist(rad, bins=25)
    plt.show()


def find_density_and_speed(event: datetime, probe: int, distance: bool = False):
    """
    Returns density and speed of the solar wind (and more if distance is True)
    :param event: event date
    :param probe: 1 or 2 for Helios 1 or 2
    :param distance: if True, also returns the distance from the sun
    :return:
    """
    start_analysis = event - timedelta(hours=1)
    imported_data = HeliosData(start_date=start_analysis.strftime('%d/%m/%Y'), start_hour=start_analysis.hour,
                               duration=2, probe=probe)
    imported_data.data.dropna(inplace=True)
    imported_data.create_processed_column('vp_magnitude')
    imported_data.create_processed_column('b_magnitude')
    data = imported_data.data.loc[event - timedelta(minutes=5):event + timedelta(minutes=5)]
    density = np.mean(data['n_p'].values)
    speed = np.mean(data['vp_magnitude'].values)
    print(event, speed, density, np.mean(data['r_sun'].values))
    if distance:
        return density, speed, np.mean(data['r_sun'].values), np.mean(data['Tp_perp'].values), np.mean(
            data['Tp_par'].values), (np.mean(2 * data['Tp_perp'].values) + np.mean(data['Tp_par'].values)) / 3, np.mean(
            data['b_magnitude'])
    else:
        return density, speed


def find_event_duration(event: datetime, probe: int) ->float:
    """
    Finds the start and end of the event by looking at changes in temperature
    :param event: time and date of reconnection
    :param probe: 1 or 2 for Helios 1 or 2
    :return: duration of the event
    """
    start_analysis = event - timedelta(hours=1)
    imported_data = HeliosData(start_date=start_analysis.strftime('%d/%m/%Y'), start_hour=start_analysis.hour,
                               duration=2, probe=probe)
    imported_data.data.dropna(inplace=True)
    data = imported_data.data.loc[event - timedelta(minutes=4): event + timedelta(minutes=4)]
    duration = []
    perp_outliers = get_outliers(get_derivative(data['Tp_perp']), standard_deviations=1.5, reference='median')
    par_outliers = get_outliers(get_derivative(data['Tp_par']), standard_deviations=1.5, reference='median')
    for n in range(len(perp_outliers)):
        if not np.isnan(perp_outliers[n]) and not np.isnan(par_outliers[n]):
            if event - timedelta(minutes=2) < perp_outliers.index[n] < event + timedelta(minutes=2):
                duration.append(perp_outliers.index[n])
    if len(duration) <= 1:
        event_duration = 2
    else:
        event_duration = (duration[-1] - duration[0]).total_seconds() / 60
        print('DURATION', event_duration)
    return event_duration


def find_data_gaps(probe, start_time: str, end_time: str) -> list:
    start = datetime.strptime(start_time, '%d/%m/%Y')
    start, end = datetime(start.year, start.month, 1), datetime.strptime(end_time, '%d/%m/%Y')
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
        days_of_year = []
        for days in range(int((date2 - date1).total_seconds() / (3600 * 24))):
            days_of_year.append(datetime.strftime(date1, '%j'))
            date1 = date1 + timedelta(days=1)
        missing = 0
        for doy in days_of_year:
            if not 'h' + str(probe) + '_' + str(year) + '_' + str(doy) + '_corefit.csv' in (np.array(fls).flatten()):
                missing += 1
        if missing > 25:
            missing_data.append([int(year), int(month)])
        date = date + interval
    return missing_data


if __name__ == '__main__':
    space_probe = 1
    probe_start_time = '15/12/1974'
    probe_end_time = '15/08/1984'
    # space_probe = 2
    # probe_start_time = '20/01/1976'
    # probe_end_time = '01/10/1979'
    planet_to_plot = 'Earth'
    dates = get_events_dates(space_probe)
    spacecraft_orbiter = get_orbiter(space_probe, probe_start_time, probe_end_time)
    missing_dates = find_data_gaps(space_probe, probe_start_time, probe_end_time)
    stats = time_stats(dates, mode='monthly')
    plot_hist_dist(spacecraft_orbiter, space_probe, stats, planet_to_plot, dates, missing_dates)
    # plot_hist_dist(spacecraft_orbiter, space_probe, stats, None, dates, missing_dates, plot_sun=True)
    histogram_speed_density()
