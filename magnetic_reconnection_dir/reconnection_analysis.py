from datetime import timedelta, datetime
from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import linregress

from data_handler.data_importer.helios_data import HeliosData
from data_handler.imported_data_plotter import plot_imported_data
from data_handler.utils.column_processing import get_outliers, get_derivative
from magnetic_reconnection_dir.csv_utils import get_dates_from_csv, create_events_list_from_csv_files
from magnetic_reconnection_dir.lmn_coordinates import plot_lmn
from magnetic_reconnection_dir.mva_analysis import hybrid_mva

proton_mass = 1.67 * 10e-27
mu_0 = np.pi * 4e-7
electron_charge = 1.6e-19
k_b = 1.38e-23


def temperature_analysis(events: List[List[Union[datetime, int]]]):
    satisfied_test = 0
    use_2_b = True
    print(len(events))
    total_t, par_t, perp_t, t_diff = [], [], [], []
    shear, small_shear, big_shear, medium_shear = get_shear_angle(events)
    for event, probe in events:
        print(event, probe)
        try:
            start = event - timedelta(hours=2)
            imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=4,
                                       probe=probe)
            imported_data.data.dropna(inplace=True)
            radius = imported_data.data.loc[event - timedelta(minutes=4):event, 'r_sun'][0]
            duration, event_start, event_end = find_intervals(imported_data, event)
            left_interval_end, right_interval_start = event_start, event_end
            left_interval_start = event_start - timedelta(minutes=5)
            right_interval_end = event_end + timedelta(minutes=5)

            b_l_left, b_l_right, n_left, n_right, L, M, N = get_n_b(event, probe, imported_data, left_interval_start,
                                                                    left_interval_end, right_interval_start,
                                                                    right_interval_end)

            if use_2_b:
                b_l, n = [b_l_left, b_l_right], [n_left, n_right]
            else:
                b_l, n = [(b_l_left + b_l_right) / 2], [(n_left + n_right) / 2]

            delta_t, dt_perp, dt_par, other_dt_total = find_temperature(imported_data, b_l, n, left_interval_start,
                                                                        left_interval_end,
                                                                        right_interval_start, right_interval_end)
            predicted_increase, alfven_speed = find_predicted_temperature(b_l, n)
            if 0.8 * delta_t <= predicted_increase * 0.13 <= 1.2 * delta_t:
                satisfied_test += 1
            if radius < 0.5:
                if [event, probe] in small_shear:
                    s = 'small'
                elif [event, probe] in big_shear:
                    s = 'big'
                else:
                    s = 'medium'
                print('small radius', radius, s)

            if delta_t > 15:
                print('DELTA T > 15', delta_t, event, radius)
            if delta_t < 0:
                print('delta t smaller than 0 ', delta_t, event, radius)

            if [event, probe] in small_shear:
                total_t.append([predicted_increase, delta_t, 'r', 'small shear'])
                par_t.append([predicted_increase, dt_par, 'r', 'small shear '])
                perp_t.append([predicted_increase, dt_perp, 'r', 'small shear '])
                t_diff.append([dt_par, dt_perp, 'r', 'small shear '])
            elif [event, probe] in big_shear:
                total_t.append([predicted_increase, delta_t, 'b', 'big shear'])
                par_t.append([predicted_increase, dt_par, 'b', 'big shear'])
                perp_t.append([predicted_increase, dt_perp, 'b', 'big shear'])
                t_diff.append([dt_par, dt_perp, 'b', 'big shear'])
            else:
                total_t.append([predicted_increase, delta_t, 'g', 'medium shear'])
                par_t.append([predicted_increase, dt_par, 'g', 'medium shear'])
                perp_t.append([predicted_increase, dt_perp, 'g', 'medium shear'])
                t_diff.append([dt_par, dt_perp, 'g', 'medium shear'])
        except ValueError:
            print('value error')
    print('satisfied test: ', satisfied_test)

    slopes = plot_relations([[total_t, 'Proton temperature change versus ' + r'$mv^2$'],
                             [par_t, 'Parallel proton temperature change versus ' + r'$mv^2$'],
                             [perp_t, 'Perpendicular proton temperature change versus ' + r'$mv^2$'],
                             [t_diff, 'Perpendicular versus parallel proton temperature changes']], 0.13)
    return slopes


def plot_relations(related_lists: List[list], slope=None):
    slopes = []
    for n in range(len(related_lists)):
        fig = plt.figure(n + 1)
        a = [x[0] for x in related_lists[n][0] if not np.isnan(x[0]) and not np.isnan(x[1])]
        b = [y[1] for y in related_lists[n][0] if not np.isnan(y[0]) and not np.isnan(y[1])]
        color = [c[2] for c in related_lists[n][0] if not np.isnan(c[0]) and not np.isnan(c[1])]
        # print(a, b)
        print(linregress(a, b))
        print(np.median(np.array(b) / np.array(a)))
        slope_linreg, intercept, rvalue, pvalue, stderr = linregress(a, b)
        slopes.append(slope_linreg)
        # print(np.polyfit(a, b, 1))
        plt.scatter(a, b, c=color, marker='+')
        plt.title(related_lists[n][1])
        if slope is not None:
            plt.plot([np.min(a), np.max(a)], [slope * np.min(a), slope * np.max(a)],
                     label='Expected gradient: ' + str(slope))
        plt.plot([np.min(a), np.max(a)], [slope_linreg * np.min(a), slope_linreg * np.max(a)],
                 label='Calculated gradient: ' + str(np.float16(slope_linreg)))
        gradient_legend = plt.legend(loc=4)
        plt.gca().add_artist(gradient_legend)
        if related_lists[n][1] == 'Perpendicular versus parallel proton temperature changes':
            plt.xlabel(r'$\Delta T_{par}$' + ' (eV)')
            plt.ylabel(r'$\Delta T_{perp}$' + ' (eV)')
        else:
            plt.xlabel(r'$mv^2$' + ' (eV)')
            plt.ylabel(r'$\Delta$' + 'T (eV)')
        blue_cross = mlines.Line2D([], [], color='blue', marker='+', linestyle='None',
                                   label='High shear angle ' + r'($\theta > 135\degree $)')
        red_cross = mlines.Line2D([], [], color='red', marker='+', linestyle='None',
                                  label='Low shear angle ' + r'($\theta < 90\degree $)')
        green_cross = mlines.Line2D([], [], color='green', marker='+', linestyle='None',
                                    label='Medium shear angle ' + r'($90\degree < \theta < 135\degree $)')

        plt.legend(handles=[blue_cross, red_cross, green_cross], loc=2)

        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    return slopes


def find_predicted_temperature(b_l: List, n: List):
    """
    Finds the predicted change from the alfven speep
    :param b_l: B in the L direction
    :param n: number density of the protons
    :return:
    """
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
    par_outliers = get_outliers(get_derivative(imported_data.data['Tp_par']), standard_deviations=1.5,
                                reference='median')
    for n in range(len(perp_outliers)):
        if not np.isnan(perp_outliers[n]) and not np.isnan(par_outliers[n]):
            if event - timedelta(minutes=2) < perp_outliers.index[n] < event + timedelta(minutes=2):
                duration.append(perp_outliers.index[n])
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
    return event_duration, event_start, event_end


def find_temperature(imported_data: HeliosData, b_l: List, n: List, left_interval_start: datetime,
                     left_interval_end: datetime, right_interval_start: datetime, right_interval_end: datetime):
    """
    Finds the inflow and exaust temperatures in order to find delta t
    :param imported_data: ImportedData
    :param b_l: B in the L direction
    :param n: number density of protons
    :param left_interval_start: start of the left interval
    :param left_interval_end: end of the left interval
    :param right_interval_start: start of the right interval
    :param right_interval_end: end of the right interval
    :return:
    """
    perpendicular_temperature, parallel_temperature = imported_data.data['Tp_perp'], imported_data.data['Tp_par']
    total_temperature = (2 * perpendicular_temperature + parallel_temperature) / 3

    def kelvin_to_ev(temperature: float):
        return temperature * k_b / electron_charge

    def get_inflow_temp_2b(temperature: pd.DataFrame, n: List, b_l: List):  # when we have b left and b right
        t_left = np.mean((temperature.loc[left_interval_start:left_interval_end]).values)
        t_right = np.mean((temperature.loc[right_interval_start:right_interval_end]).values)
        inflow = (n[0] * t_left / b_l[0] + n[1] * t_right / b_l[1]) / (n[0] / b_l[0] + n[1] / b_l[1])
        return inflow

    def get_inflow_temp_1b(temperature: pd.DataFrame, n: List, b_l: List):  # when we use only one b
        t_left = np.mean((temperature.loc[left_interval_start:left_interval_end]).values)
        t_right = np.mean((temperature.loc[right_interval_start:right_interval_end]).values)
        inflow = (t_left + t_right) / 2
        return inflow

    def get_delta_t(temperature: pd.DataFrame, t_inflow: float):
        t_exhaust = np.percentile((temperature.loc[left_interval_end:right_interval_start]).values, 90)
        print(t_exhaust, np.max((temperature.loc[left_interval_end:right_interval_start]).values))
        return np.abs(t_exhaust - t_inflow) * np.sign(t_exhaust - t_inflow)

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
    """
    :param event: event date
    :param probe: 1 or 2 for Helios 1 or 2
    :param imported_data: ImportedData
    :param left_interval_start: start of left interval
    :param left_interval_end: end of left interval
    :param right_interval_start: start of right interval
    :param right_interval_end: end of right interval
    :return:
    """
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


def get_shear_angle(events_list: List[List[Union[datetime, int]]]):
    """
    Finds the shear angle of events
    :param events_list: list of events to be analysed
    :return: shear angles, and lists of events with low, medium and high shear angles
    """
    shear = []
    small_shear, big_shear, medium_shear = [], [], []
    for event, probe in events_list:
        print(event)
        start = event - timedelta(hours=1)
        imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=2,
                                   probe=probe)
        imported_data.data.dropna(inplace=True)
        duration, event_start, event_end = find_intervals(imported_data, event)
        left_interval_end, right_interval_start = event_start, event_end
        left_interval_start = event_start - timedelta(minutes=5)
        right_interval_end = event_end + timedelta(minutes=5)

        b_left = (np.array([np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'Bx']).values),
                            np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'By']).values),
                            np.mean((imported_data.data.loc[left_interval_start:left_interval_end, 'Bz']).values)]))
        b_right = (np.array([np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'Bx']).values),
                             np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'By']).values),
                             np.mean((imported_data.data.loc[right_interval_start: right_interval_end, 'Bz']).values)]))
        br_mag = np.sqrt(b_left[0] ** 2 + b_left[1] ** 2 + b_left[2] ** 2)
        bl_mag = np.sqrt(b_right[0] ** 2 + b_right[1] ** 2 + b_right[2] ** 2)
        theta = np.arccos((np.dot(b_right, b_left) / (bl_mag * br_mag)))
        theta = np.degrees(theta)
        if not np.isnan(theta):
            shear.append(theta)
        if theta <= 90:
            small_shear.append([event, probe])
        elif theta > 135:
            big_shear.append([event, probe])
        else:
            medium_shear.append([event, probe])
    print('shear', shear)
    # plt.hist(shear, bins=15)
    # plt.show()
    return shear, small_shear, big_shear, medium_shear


if __name__ == '__main__':
    # events1 = get_dates_from_csv(filename='helios1_magrec2.csv', probe=1)
    # events2 = get_dates_from_csv(filename='helios2_magrec2.csv', probe=2)
    # events_to_analyse = events1 + events2

    events_to_analyse = create_events_list_from_csv_files([['helios1_magrec2.csv', 1], ['helios1mag_rec3.csv', 1]])
    events_to_analyse = events_to_analyse + create_events_list_from_csv_files(
        [['helios2_magrec2.csv', 2], ['helios2mag_rec3.csv', 2]])
    temperature_analysis(events=events_to_analyse)
    # shear_angle, small_shear_angle, big_shear_angle, medium_shear_angle = get_shear_angle(events_to_analyse)
    # print('small shear angle', small_shear_angle)
    # temperature_analysis(small_shear_angle)
    # print('big shear angle', big_shear_angle)
    # temperature_analysis(big_shear_angle)
    # print('medium shear angle', medium_shear_angle)
    # temperature_analysis(medium_shear_angle)
