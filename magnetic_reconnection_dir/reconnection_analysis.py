import csv
from datetime import timedelta, datetime
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from data_handler.data_importer.helios_data import HeliosData
from data_handler.imported_data_plotter import plot_imported_data
from data_handler.utils.column_processing import get_outliers
from magnetic_reconnection_dir.lmn_coordinates import hybrid_mva

proton_mass = 1.67 * 10e-27
mu_0 = np.pi * 4e-7
electron_charge = 1.6e-19
k_b = 1.38e-23


def temperature_analysis(events: List[List[Union[datetime, int]]]):
    time_around = 0.5
    time_outside = 2
    satisfied_test = 0
    too_big = 0
    use_2_b = True
    print(len(events))
    total_t, par_t, perp_t = [], [], []
    for event, probe in events:
        print(event, probe)
        try:
            left_interval_start = event - timedelta(minutes=5)
            left_interval_end = event - timedelta(minutes=2)
            right_interval_start = event + timedelta(minutes=0.5)
            right_interval_end = event + timedelta(minutes=3.5)

            start = event - timedelta(hours=1)
            imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=2,
                                       probe=probe)
            duration, event_start, event_end = find_intervals(imported_data, event)
            left_interval_end = event_start
            left_interval_start = event_start - timedelta(minutes=duration)
            right_interval_start = event_end
            right_interval_end = event_end + timedelta(minutes=duration)

            # plot_imported_data(imported_data, event_date=event, boundaries=[left_interval_end, right_interval_start])
            imported_data.data.dropna(inplace=True)
            imported_data.create_processed_column('vp_magnitude')
            imported_data.create_processed_column('b_magnitude')
            b_l_left, b_l_right, n_left, n_right = get_n_b(event, probe, imported_data, left_interval_start,
                                                           left_interval_end, right_interval_start, right_interval_end)

            b_left = np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'b_magnitude'])
            b_right = np.mean(imported_data.data.loc[right_interval_start:right_interval_end, 'b_magnitude'])
            if use_2_b:
                b_l = [b_l_left, b_l_right]
                b = [b_left, b_right]
                n = [n_left, n_right]
            else:
                b_l = [(b_l_left + b_l_right) / 2]
                b = [(b_left + b_right) / 2]
                n = [(n_left + n_right) / 2]

            delta_t, dt_perp, dt_par = find_temperature(imported_data, b_l, n, left_interval_start, left_interval_end,
                                                        right_interval_start, right_interval_end)
            predicted_increase, pred, speed, alvfen_speed = find_predicted_temperature(imported_data, b_l, n,
                                                                                       left_interval_end,
                                                                                       right_interval_start)
            n_inside = np.mean((imported_data.data.loc[left_interval_end:right_interval_start, 'n_p']).values)
            if 0.8 * delta_t <= pred * 0.022342463657490763 <= 1.2 * delta_t:
                # print(event)
                # print(speed, alvfen_speed)
                # print(predicted_increase, pred)
                satisfied_test += 1
            if dt_perp > 100000*  dt_par:
                print('PERPENDICULAR  BIGGER: ', dt_perp, dt_par, event,
                      imported_data.data.loc[event - timedelta(minutes=4):event, 'r_sun'][0])
                # plot_imported_data(imported_data, event_date=event,
                #                    boundaries=[left_interval_end, right_interval_start])
            elif delta_t > 50:
                print('> 600: ', event, imported_data.data.loc[event - timedelta(minutes=1):event, 'r_sun'][0])
                too_big += 1
            elif dt_perp > 100 or dt_par > 100:
                print('perp or par > 40: ', event,
                      imported_data.data.loc[event - timedelta(minutes=4):event, 'r_sun'][0])
                too_big += 1

            # elif n_inside > 1.1*n_right and n_inside > 1.1*n_left:
            #     print('N TOO BIG: ', event, n_inside, n_right, imported_data.data.loc[event - timedelta(minutes=1):event, 'r_sun'][0])
            else:
                total_t.append([pred, delta_t])
                par_t.append([pred, dt_par])
                perp_t.append(([pred, dt_perp]))
        except ValueError:
            print('value error')
    print('satisfied test: ', satisfied_test)
    print('too big values: ', too_big)

    slopes = plot_relations([total_t, par_t, perp_t])
    return slopes


def plot_relations(related_lists: List[list], slope=None):
    slopes = []
    for n in range(len(related_lists)):
        fig = plt.figure(n + 1)
        a = [x[0] for x in related_lists[n] if not np.isnan(x[0]) and not np.isnan(x[1])]
        b = [y[1] for y in related_lists[n] if not np.isnan(y[0]) and not np.isnan(y[1])]
        # print(a, b)
        print(linregress(a, b))
        print(np.median(np.array(b) / np.array(a)))
        slope_linreg, intercept, rvalue, pvalue, stderr = linregress(a, b)
        slopes.append(slope_linreg)
        # print(np.polyfit(a, b, 1))
        plt.scatter(a, b)
        if slope is not None:
            plt.plot([0, np.max(a)], [0, slope * np.max(a)])
        plt.plot([0, np.max(a)], [0, slope_linreg * np.max(a)])
        plt.xlabel('mv**2')
        plt.ylabel('delta_t')
    plt.show()
    return slopes


def find_predicted_temperature(imported_data, b_l: List, n: List, left_interval_end, right_interval_start):
    speed = np.mean(imported_data.data.loc[left_interval_end:right_interval_start, 'vp_magnitude']) * 10 ** 3
    predicted_increase = (proton_mass * speed ** 2) / electron_charge
    if len(b_l) == 1 and len(n) == 1:
        alvfen_speed = b_l[0] * 10 ** (-9) / np.sqrt(n[0] * 10 ** 6 * proton_mass * mu_0)  # b in nT, n in cm^-3
        pred = (proton_mass * alvfen_speed ** 2) / electron_charge
    elif len(b_l) == 2 and len(n) == 2:
        alvfen_speed = np.sqrt(((b_l[0] + b_l[1]) * b_l[0] * b_l[1] * 10 ** (-27)) / (
                mu_0 * 10 ** 6 * 10 ** (-9) * proton_mass * (b_l[0] * n[1] + b_l[1] * n[0])))
        pred = (proton_mass * alvfen_speed ** 2) / electron_charge
    else:
        raise ValueError('b_l and n must have the same length between 1 and 2')
    # print(b_l, alvfen_speed/10**3, pred)
    return predicted_increase, pred, speed, alvfen_speed


def find_intervals(imported_data, event):
    """
    Finds the start and end of the event by looking at changes in temperature
    :param imported_data: ImportedData
    :param event: time and date of reconnection
    :return:
    """
    duration = []
    perp_outliers = get_outliers(imported_data.data['Tp_perp'], standard_deviations=2, reference='median')
    par_outliers = get_outliers(imported_data.data['Tp_par'], standard_deviations=2, reference='median')
    for n in range(len(perp_outliers)):
        if not np.isnan(perp_outliers[n]) and not np.isnan(par_outliers[n]):
            if event - timedelta(minutes=4) < perp_outliers.index[n] < event + timedelta(minutes=4):
                duration.append(perp_outliers.index[n])
    print(len(duration))
    if len(duration) <= 1:
        event_duration = 2
        if len(duration)==0:
            event_start = event - timedelta(minutes=event_duration/2)
            event_end = event + timedelta(minutes=event_duration/2)
        else:
            event_start = duration[0] - timedelta(minutes=event_duration / 2)
            event_end = duration[0] + timedelta(minutes=event_duration / 2)
    else:
        event_duration = (duration[-1] - duration[0]).total_seconds()/60

        event_start = duration[0]
        event_end = duration[-1]
        print('DURATION', event_duration, event_start, event_end)
    return event_duration, event_start, event_end


def find_temperature(imported_data: HeliosData, b_l, n, left_interval_start, left_interval_end, right_interval_start,
                     right_interval_end):
    perpendicular_temperature = imported_data.data['Tp_perp']
    parallel_temperature = imported_data.data['Tp_par']
    total_temperature = (2 * perpendicular_temperature + parallel_temperature) / 3

    def kelvin_to_ev(temperature: float):
        temperature = temperature * k_b / electron_charge
        return temperature

    def get_inflow_temp_2b(temperature: pd.DataFrame, n: List, b_l: List):
        t_left = temperature.loc[left_interval_start:left_interval_end]
        t_left = np.mean(t_left.values)
        t_right = temperature.loc[right_interval_start:right_interval_end]
        t_right = np.mean(t_right.values)
        inflow = (n[0] * t_left / b_l[0] + n[1] * t_right / b_l[1]) / (n[0] / b_l[0] + n[1] / b_l[1])
        return inflow

    def get_inflow_temp_1b(temperature: pd.DataFrame, n: List, b_l: List):
        t_left = temperature.loc[left_interval_start:left_interval_end]
        t_left = np.mean(t_left.values)
        t_right = temperature.loc[right_interval_start:right_interval_end]
        t_right = np.mean(t_right.values)
        inflow = (t_left + t_right) / 2
        return inflow

    def get_delta_t(temperature: pd.DataFrame, t_inflow: float):
        t_exhaust = temperature.loc[left_interval_end:right_interval_start]
        t_exhaust = np.mean(t_exhaust.values)
        delta_t = np.abs(t_inflow - t_exhaust)
        return delta_t

    if len(b_l) == 2:
        total_inflow = get_inflow_temp_2b(total_temperature, n, b_l)
        perp_inflow = get_inflow_temp_2b(perpendicular_temperature, n, b_l)
        par_inflow = get_inflow_temp_2b(parallel_temperature, n, b_l)
    else:
        total_inflow = get_inflow_temp_1b(total_temperature, n, b_l)
        perp_inflow = get_inflow_temp_1b(perpendicular_temperature, n, b_l)
        par_inflow = get_inflow_temp_1b(parallel_temperature, n, b_l)

    delta_t_total = get_delta_t(total_temperature, total_inflow)
    delta_t_perp = get_delta_t(perpendicular_temperature, perp_inflow)
    delta_t_par = get_delta_t(parallel_temperature, par_inflow)

    print(total_inflow, delta_t_total)
    print('TEMPERATURE: ', kelvin_to_ev(delta_t_total), kelvin_to_ev(delta_t_perp), kelvin_to_ev(delta_t_par))
    return kelvin_to_ev(delta_t_total), kelvin_to_ev(delta_t_perp), kelvin_to_ev(delta_t_par)


def get_n_b(event: datetime, probe: int, imported_data: HeliosData, left_interval_start, left_interval_end,
            right_interval_start, right_interval_end):
    L, M, N = hybrid_mva(event, probe, 5, 1)
    b_left = np.abs(np.array([np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'Bx']).values),
                              np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'By']).values),
                              np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'Bz']).values)]))
    b_right = np.abs(np.array([np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'Bx']).values),
                               np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'By']).values),
                               np.mean(
                                   (imported_data.data.loc[right_interval_start: right_interval_end, 'Bz']).values)]))
    b_l_left, b_l_right = np.dot(b_left, L), np.dot(b_right, L)
    n_left = np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'n_p']).values)
    n_right = np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'n_p']).values)

    return b_l_left, b_l_right, n_left, n_right


def get_data(file_name: str, probe: int = 1):
    events_list = []
    with open(file_name) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year, month, day = np.int(row['year']), np.int(row['month']), np.int(row['day'])
            hours, minutes, seconds = np.int(row['hours']), np.int(row['minutes']), np.int(row['seconds'])
            events_list.append([datetime(year, month, day, hours, minutes, seconds), probe])
    return events_list


if __name__ == '__main__':
    events1 = get_data('helios1_magrec2.csv', probe=1)
    events2 = get_data('helios2_magrec2.csv', probe=2)
    # temperature_analysis([[datetime(1978, 4, 22, 10, 31), 2]])  ## weird event, clearly not fitting!!! close to sun
    # temperature_analysis([[datetime(1980, 5, 29, 15, 39), 1]])  ## weird event, clearly not fitting!!! close to sun
    # temperature_analysis([[datetime(1976, 5, 4, 1, 58), 1]])  ## weird event, clearly not fitting!!! close to sun
    temperature_analysis(events=events1 + events2)
