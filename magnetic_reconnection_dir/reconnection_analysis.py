from datetime import timedelta, datetime
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from data_handler.data_importer.helios_data import HeliosData
from data_handler.imported_data_plotter import plot_imported_data
from data_handler.utils.column_processing import get_outliers, get_derivative
from magnetic_reconnection_dir.csv_utils import get_dates_from_csv
from magnetic_reconnection_dir.mva_analysis import hybrid_mva

proton_mass = 1.67 * 10e-27
mu_0 = np.pi * 4e-7
electron_charge = 1.6e-19
k_b = 1.38e-23


def temperature_analysis(events: List[List[Union[datetime, int]]]):
    satisfied_test = 0
    too_big = 0
    use_2_b = True
    print(len(events))
    total_t, par_t, perp_t, other_total_t = [], [], [], []
    for event, probe in events:
        print(event, probe)
        try:
            start = event - timedelta(hours=2)
            imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=4,
                                       probe=probe)
            imported_data.data.dropna(inplace=True)
            duration, event_start, event_end = find_intervals(imported_data, event)
            left_interval_end, right_interval_start = event_start, event_end
            left_interval_start = event_start - timedelta(minutes=5)
            right_interval_end = event_end + timedelta(minutes=5)

            b_l_left, b_l_right, n_left, n_right, L, M, N = get_n_b(event, probe, imported_data, left_interval_start,
                                                                    left_interval_end, right_interval_start,
                                                                    right_interval_end)

            if use_2_b:
                b_l = [b_l_left, b_l_right]
                n = [n_left, n_right]
            else:
                b_l = [(b_l_left + b_l_right) / 2]
                n = [(n_left + n_right) / 2]

            delta_t, dt_perp, dt_par, other_dt_total = find_temperature(imported_data, b_l, n, left_interval_start,
                                                                        left_interval_end,
                                                                        right_interval_start, right_interval_end)
            predicted_increase, alfven_speed = find_predicted_temperature(b_l, n)
            # plot_lmn(imported_data=imported_data, L=L, M=M, N=N, event_date=event, boundaries=[left_interval_end, right_interval_start])
            n_inside = np.mean((imported_data.data.loc[left_interval_end:right_interval_start, 'n_p']).values)
            if 0.8 * delta_t <= predicted_increase * 0.13 <= 1.2 * delta_t:
                # print(event, alfven_speed, predicted_increase)
                satisfied_test += 1
            if dt_perp > 100000 * dt_par:
                print('PERPENDICULAR  BIGGER: ', dt_perp, dt_par, event,
                      imported_data.data.loc[event - timedelta(minutes=4):event, 'r_sun'][0])
                # plot_imported_data(imported_data, event_date=event,
                #                    boundaries=[left_interval_end, right_interval_start])
            elif delta_t > 150:
                print('> 150: ', event, imported_data.data.loc[event - timedelta(minutes=1):event, 'r_sun'][0])
                too_big += 1
            # elif delta_t < 0.1:
            #     print('this event might have too small an increase in temperature')
                # plot_lmn(imported_data=imported_data, L=L, M=M, N=N, event_date=event,
                #          boundaries=[left_interval_end, right_interval_start])

            elif dt_perp > 60 or dt_par > 60:
                print('perp or par > 60: ', event,
                      imported_data.data.loc[event - timedelta(minutes=4):event, 'r_sun'][0])
                too_big += 1

            # elif n_inside > 1.1*n_right and n_inside > 1.1*n_left:
            #     print('N TOO BIG: ', event, n_inside, n_right, imported_data.data.loc[event - timedelta(minutes=1):event, 'r_sun'][0])
            else:
                total_t.append([predicted_increase, delta_t])
                par_t.append([predicted_increase, dt_par])
                perp_t.append([predicted_increase, dt_perp])
                other_total_t.append([predicted_increase, other_dt_total])
        except ValueError:
            print('value error')
    print('satisfied test: ', satisfied_test)
    print('too big values: ', too_big)

    slopes = plot_relations([total_t, par_t, perp_t], 0.13)
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
            plt.plot([np.min(a), np.max(a)], [slope * np.min(a), slope * np.max(a)])
        plt.plot([np.min(a), np.max(a)], [slope_linreg * np.min(a), slope_linreg * np.max(a)])
        plt.xlabel('mv**2')
        plt.ylabel('delta_t')
        plt.xscale('log')
        plt.yscale('log')
    plt.show()
    return slopes


def find_predicted_temperature(b_l: List, n: List):
    if len(b_l) == 1 and len(n) == 1:
        alfven_speed = b_l[0] * 10 ** (-9) / np.sqrt(n[0] * 10 ** 6 * proton_mass * mu_0)  # b in nT, n in cm^-3
        predicted_increase = (proton_mass * alfven_speed ** 2) / electron_charge
    elif len(b_l) == 2 and len(n) == 2:
        alfven_speed = np.sqrt(((b_l[0] + b_l[1]) * b_l[0] * b_l[1] * 10 ** (-27)) / (
                mu_0 * 10 ** 6 * 10 ** (-9) * proton_mass * (b_l[0] * n[1] + b_l[1] * n[0])))
        predicted_increase = (proton_mass * alfven_speed ** 2) / electron_charge
    else:
        raise ValueError('b_l and n must have the same length between 1 and 2')
    print(b_l, alfven_speed / 10 ** 3, predicted_increase)
    return predicted_increase, alfven_speed


def find_intervals(imported_data: HeliosData, event: datetime):
    """
    Finds the start and end of the event by looking at changes in temperature
    :param imported_data: ImportedData
    :param event: time and date of reconnection
    :return:
    """
    duration = []
    perp_outliers = get_outliers(get_derivative(imported_data.data['Tp_perp']), standard_deviations=1.5,
                                 reference='median')
    par_outliers = get_outliers(get_derivative(imported_data.data['Tp_par']), standard_deviations=1.5, reference='median')
    for n in range(len(perp_outliers)):
        if not np.isnan(perp_outliers[n]) and not np.isnan(par_outliers[n]):
            if event - timedelta(minutes=2) < perp_outliers.index[n] < event + timedelta(minutes=2):
                duration.append(perp_outliers.index[n])
    print(len(duration))
    if len(duration) <= 1:
        event_duration = 2
        if len(duration) == 0:
            event_start = event - timedelta(minutes=event_duration / 2)
            event_end = event + timedelta(minutes=event_duration / 2)
        else:
            event_start = duration[0] - timedelta(minutes=event_duration / 2)
            event_end = duration[0] + timedelta(minutes=event_duration / 2)
    else:
        event_duration = (duration[-1] - duration[0]).total_seconds() / 60

        event_start = duration[0]
        event_end = duration[-1]
        print('DURATION', event_duration, event_start, event_end)
    return event_duration, event_start, event_end


def find_temperature(imported_data: HeliosData, b_l: List, n: List, left_interval_start: datetime,
                     left_interval_end: datetime, right_interval_start: datetime, right_interval_end: datetime):
    perpendicular_temperature, parallel_temperature = imported_data.data['Tp_perp'], imported_data.data['Tp_par']
    total_temperature = (2 * perpendicular_temperature + parallel_temperature) / 3

    def kelvin_to_ev(temperature: float):
        return temperature * k_b / electron_charge

    def get_inflow_temp_2b(temperature: pd.DataFrame, n: List, b_l: List):
        t_left = np.mean((temperature.loc[left_interval_start:left_interval_end]).values)
        t_right = np.mean((temperature.loc[right_interval_start:right_interval_end]).values)
        inflow = (n[0] * t_left / b_l[0] + n[1] * t_right / b_l[1]) / (n[0] / b_l[0] + n[1] / b_l[1])
        return inflow

    def get_inflow_temp_1b(temperature: pd.DataFrame, n: List, b_l: List):
        t_left = np.mean((temperature.loc[left_interval_start:left_interval_end]).values)
        t_right = np.mean((temperature.loc[right_interval_start:right_interval_end]).values)
        inflow = (t_left + t_right) / 2
        return inflow

    def get_delta_t(temperature: pd.DataFrame, t_inflow: float):
        t_exhaust = np.max((temperature.loc[left_interval_end:right_interval_start]).values)
        return np.abs(t_inflow - t_exhaust)

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

    print(total_inflow, delta_t_total, delta_t_perp, delta_t_par, (2 * delta_t_perp + delta_t_par) / 3)
    print('TEMPERATURE: ', kelvin_to_ev(delta_t_total), kelvin_to_ev(delta_t_perp), kelvin_to_ev(delta_t_par))
    return kelvin_to_ev(delta_t_total), kelvin_to_ev(delta_t_perp), kelvin_to_ev(delta_t_par), (
            2 * kelvin_to_ev(delta_t_perp) + kelvin_to_ev(delta_t_par)) / 3


def get_n_b(event: datetime, probe: int, imported_data: HeliosData, left_interval_start: datetime,
            left_interval_end: datetime, right_interval_start: datetime, right_interval_end: datetime):
    L, M, N = hybrid_mva(event, probe, outside_interval=5, inside_interval=1, mva_interval=10)
    b_left = (np.array([np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'Bx']).values),
                        np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'By']).values),
                        np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'Bz']).values)]))
    b_right = (np.array([np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'Bx']).values),
                         np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'By']).values),
                         np.mean(
                             (imported_data.data.loc[right_interval_start: right_interval_end, 'Bz']).values)]))
    print(b_left, b_right)
    b_l_left, b_l_right = np.abs(np.dot(b_left, L)), np.abs(np.dot(b_right, L))
    n_left = np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'n_p']).values)
    n_right = np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'n_p']).values)
    print(b_l_left, b_l_right, n_left, n_right)
    return b_l_left, b_l_right, n_left, n_right, L, M, N


if __name__ == '__main__':
    events1 = get_dates_from_csv(filename='helios1_magrec2.csv', probe=1)
    events2 = get_dates_from_csv(filename='helios2_magrec2.csv', probe=2)
    events22 = get_dates_from_csv(filename='helios2mag_rec3.csv', probe=2)
    events12 = get_dates_from_csv(filename='helios1mag_rec3.csv', probe=1)
    # temperature_analysis([[datetime(1978, 4, 22, 10, 31), 2]])  ## weird event, clearly not fitting!!! close to sun
    # temperature_analysis([[datetime(1980, 5, 29, 15, 39), 1]])  ## weird event, clearly not fitting!!! close to sun
    # temperature_analysis([[datetime(1976, 5, 4, 1, 58), 1]])  ## weird event, clearly not fitting!!! close to sun
    events_to_analyse = events1 + events2
    for event in events22:
        if event not in events2:
            print(event)
            events_to_analyse.append(event)
    for event in events12:
        if event not in events1:
            print(event)
            events_to_analyse.append(event)
    temperature_analysis(events=events_to_analyse)
