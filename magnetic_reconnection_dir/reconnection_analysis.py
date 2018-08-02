import csv
from datetime import timedelta, datetime
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from data_handler.data_importer.helios_data import HeliosData
from data_handler.imported_data_plotter import plot_imported_data
from magnetic_reconnection_dir.lmn_coordinates import hybrid_mva

proton_mass = 1.67 * 10e-27
mu_0 = np.pi * 4e-7
electron_charge = 1.6e-19
k_b = 1.38e-23


def temperature_analysis(events: List[List[Union[datetime, int]]]):
    time_around = 1
    time_outside = 3
    satisfied_test = 0
    too_big = 0
    print(len(events))
    total_t, par_t, perp_t = [], [], []
    for event, probe in events:
        print(event, probe)
        try:
            left_interval_start = event - timedelta(minutes=time_outside)
            left_interval_end = event - timedelta(minutes=time_around)
            right_interval_start = event + timedelta(minutes=time_around)
            right_interval_end = event + timedelta(minutes=time_outside)

            start = event - timedelta(hours=1)
            imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=3,
                                       probe=probe)
            imported_data.create_processed_column('vp_magnitude')
            imported_data.create_processed_column('b_magnitude')
            b_l, n = get_n_b(event, probe, imported_data, left_interval_start, left_interval_end, right_interval_start,
                             right_interval_end)

            delta_t, dt_perp, dt_par = find_temperature(imported_data, left_interval_start, left_interval_end,
                                                        right_interval_start, right_interval_end)

            b = (np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'b_magnitude']) +
                 np.mean(imported_data.data.loc[right_interval_start:right_interval_end, 'b_magnitude'])) / 2
            predicted_increase, pred, speed, alvfen_speed = find_predicted_temperature(imported_data, b, n,
                                                                                       left_interval_end,
                                                                                       right_interval_start)
            if 0.8 * delta_t <= pred * 0.02 <= 1.2 * delta_t:
                print(event)
                print(speed, alvfen_speed)
                print(predicted_increase, pred)
                satisfied_test += 1
            if delta_t > 60:
                print('> 60: ', event, imported_data.data.loc[event - timedelta(minutes=1):event, 'r_sun'][0])
                too_big += 1
            elif dt_perp > 40 or dt_par > 40:
                print('perp or par > 40: ', event,
                      imported_data.data.loc[event - timedelta(minutes=1):event, 'r_sun'][0])
                too_big += 1
            else:
                total_t.append([pred, delta_t])
                par_t.append([pred, dt_par])
                perp_t.append(([pred, dt_perp]))
        except ValueError:
            print('value error')
    print('satisfied test: ', satisfied_test)
    print('too big values: ', too_big)

    slopes = plot_relations([total_t, par_t, perp_t], 0.02)
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


def find_predicted_temperature(imported_data, b_l, n, left_interval_end, right_interval_start):
    speed = np.mean(imported_data.data.loc[left_interval_end:right_interval_start, 'vp_magnitude']) * 10 ** 3
    alvfen_speed = b_l * 10 ** (-9) / np.sqrt(n * 10 ** 6 * proton_mass * mu_0)  # b in nT, n in cm^-3
    predicted_increase = (proton_mass * speed ** 2) / electron_charge
    pred = (proton_mass * alvfen_speed ** 2) / electron_charge
    return predicted_increase, pred, speed, alvfen_speed


def find_temperature(imported_data: HeliosData, left_interval_start, left_interval_end, right_interval_start,
                     right_interval_end):
    n = 1  # np.mean(imported_data.data.loc[left_interval_start: right_interval_end, 'n_p'])
    to_ev = k_b / electron_charge
    perpendicular_temperature = imported_data.data['Tp_perp'] * to_ev * n
    parallel_temperature = imported_data.data['Tp_par'] * to_ev * n
    t_perp = (np.mean(perpendicular_temperature.loc[left_interval_start:left_interval_end]) + np.mean(
        perpendicular_temperature.loc[right_interval_start:right_interval_end])) / 2
    dt_perp = np.abs(t_perp - np.mean(perpendicular_temperature.loc[left_interval_end:right_interval_start]))
    t_par = (np.mean(parallel_temperature.loc[left_interval_start:left_interval_end]) + np.mean(
        parallel_temperature.loc[right_interval_start:right_interval_end])) / 2
    dt_par = np.abs(t_par - np.mean(parallel_temperature.loc[left_interval_end:right_interval_start]))

    temp = (2 * perpendicular_temperature + parallel_temperature) / 3
    temperature = np.mean(temp.loc[left_interval_end:right_interval_start])
    t_inflow = (np.mean(temp.loc[left_interval_start:left_interval_end]) + np.mean(
        temp.loc[right_interval_start:right_interval_end])) / 2
    delta_t = np.abs(t_inflow - temperature)
    print('TEMPERATURE: ', delta_t, dt_perp, dt_par)
    return delta_t, dt_perp, dt_par


def get_n_b(event: datetime, probe: int, imported_data: HeliosData, left_interval_start, left_interval_end,
            right_interval_start, right_interval_end):
    L, M, N = hybrid_mva(event, probe)
    # print(np.dot((np.array([np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'Bx']),
    #                       np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'By']),
    #                       np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'Bz'])])), L))
    # print(np.dot((np.array([np.mean(imported_data.data.loc[right_interval_start: right_interval_end, 'Bx']),
    #                       np.mean(imported_data.data.loc[right_interval_start: right_interval_end, 'By']),
    #                       np.mean(imported_data.data.loc[right_interval_start: right_interval_end, 'Bz'])])), L))
    b = (np.abs(np.array([np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'Bx']),
                          np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'By']),
                          np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'Bz'])])) +
         np.abs(np.array([np.mean(imported_data.data.loc[right_interval_start: right_interval_end, 'Bx']),
                          np.mean(imported_data.data.loc[right_interval_start: right_interval_end, 'By']),
                          np.mean(imported_data.data.loc[right_interval_start: right_interval_end, 'Bz'])]))) / 2
    b_l = np.dot(b, L)
    n = (np.mean(imported_data.data.loc[left_interval_start:left_interval_end, 'n_p']) +
         np.mean(imported_data.data.loc[right_interval_start: right_interval_end, 'n_p'])) / 2

    return b_l, n


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
