import numpy as np
from datetime import datetime
from numpy import linalg as LA
import csv
from datetime import timedelta
import pandas as pd
from typing import List

from data_handler.data_importer.helios_data import HeliosData
from data_handler.data_importer.imported_data import ImportedData
from data_handler.imported_data_plotter import plot_imported_data, DEFAULT_PLOTTED_COLUMNS
from data_handler.utils.column_processing import get_derivative


# test data
# B = [np.array([-13.6, -24.7, 54.6]), np.array([-14.8, -24.9, 58.7]), np.array([-13.4, -17.2, 62.4]),
#          np.array([-14.0, -25.0, 43.8]), np.array([-7.1, -4.5, 33.5]), np.array([-0.9, -5.0, 44.4]),
#          np.array([-10.0, -0.4, 44.6]), np.array([-6.1, -4.8, 21.1]), np.array([1.2, 1.6, 21.1]),
#          np.array([-3.4, -3.9, 4.1]), np.array([-0.9, 1.2, 5.0]), np.array([-1.0, -1.5, 12.3]),
#          np.array([11.0, 13.2, 29.7]), np.array([19.1, 34.4, 20.1]), np.array([24.9, 50.1, 1.9]),
#          np.array([29.2, 47.1, -10.6])]


def get_b(imported_data: ImportedData, event_date, interval: int = 30) -> List[np.ndarray]:
    """
    Returns the imported data in a suitable vector form to be analysed
    :param imported_data: ImportedData
    :return: array [bx, by, bz]
    """
    B = []
    data = imported_data.data[event_date - timedelta(minutes=interval):event_date + timedelta(minutes=interval)]
    b_x, b_y, b_z = data['Bx'].values, data['By'].values, data['Bz'].values
    for n in range(len(b_x)):
        B.append(np.array([b_x[n], b_y[n], b_z[n]]))
    return B


def get_side_data(imported_data: ImportedData, event_date: datetime, outside_interval: int = 10,
                  inside_interval: int = 2):
    """
    Returns B around the reconnection event
    :param imported_data: ImportedData
    :param event_date: date of possible reconnection
    :return:
    """
    # we need B to be stable over some period of time on both sides of the exhaust
    # we take the average in that stability region, without the reconnection event (few minutes on each side)
    # hard to determine computationally, so use 10-2 intervals

    data_1 = imported_data.data[
             event_date - timedelta(minutes=outside_interval):event_date - timedelta(minutes=inside_interval)]
    data_2 = imported_data.data[
             event_date + timedelta(minutes=inside_interval):event_date + timedelta(minutes=outside_interval)]

    b_x_1, b_y_1, b_z_1 = np.mean(data_1['Bx'].values), np.mean(data_1['By'].values), np.mean(data_1['Bz'].values)
    b_x_2, b_y_2, b_z_2 = np.mean(data_2['Bx'].values), np.mean(data_2['By'].values), np.mean(data_2['Bz'].values)
    B1, B2 = np.array([b_x_1, b_y_1, b_z_1]), np.array([b_x_2, b_y_2, b_z_2])

    v_x_1, v_y_1, v_z_1 = np.mean(data_1['vp_x'].values), np.mean(data_1['vp_y'].values), np.mean(data_1['vp_z'].values)
    v_x_2, v_y_2, v_z_2 = np.mean(data_2['vp_x'].values), np.mean(data_2['vp_y'].values), np.mean(data_2['vp_z'].values)
    v1, v2 = np.array([v_x_1, v_y_1, v_z_1]), np.array([v_x_2, v_y_2, v_z_2])

    density_1, density_2 = np.mean(data_1['n_p'].values), np.mean(data_2['n_p'].values)
    T_par_1, T_perp_1 = np.mean(data_1['Tp_par'].values), np.mean(data_1['Tp_perp'].values)
    T_par_2, T_perp_2 = np.mean(data_2['Tp_par'].values), np.mean(data_2['Tp_perp'].values)

    return B1, B2, v1, v2, density_1, density_2, T_par_1, T_perp_1, T_par_2, T_perp_2


def mva(B: List[np.ndarray]):
    """
    Finds the LMN component of the new coordinates system
    :param B: field around interval that will be considered
    :return:
    """
    # we want to solve the magnetic matrix eigenvalue problem
    M_b = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for n in range(3):
        bn = np.array([b[n] for b in B])
        for m in range(3):
            bm = np.array([b[m] for b in B])

            M_b[n, m] = np.mean(bn * bm) - np.mean(bn) * np.mean(bm)

    w, v = LA.eig(M_b)
    w_max = np.argmax(w)  # maximum value gives L
    w_min = np.argmin(w)  # minimum eigenvalue gives N
    w_intermediate = np.min(np.delete([0, 1, 2], [w_min, w_max]))

    L, N, M = np.zeros(3), np.zeros(3), np.zeros(3)
    for coordinate in range(len(v[:, w_max])):
        L[coordinate] = v[:, w_max][coordinate]
        N[coordinate] = v[:, w_min][coordinate]
        M[coordinate] = v[:, w_intermediate][coordinate]

    # print(L, M, N)

    return L, M, N


def hybrid(_L: np.ndarray, B1: np.ndarray, B2: np.ndarray):  # hybrid mva necessary if eigenvalues not well resolved
    """
    Finds the other components of the new coordinates system
    :param _L: L vector found with mva
    :param B1: mean magnetic field vector from inflow region 1
    :param B2: mean magnetic field vector from inflow region 2
    :return:
    """
    cross_of_b = np.cross(B1, B2)
    N = cross_of_b / np.sqrt(cross_of_b[0] ** 2 + cross_of_b[1] ** 2 + cross_of_b[2] ** 2)  # normalised vector
    cross_n_l = np.cross(N, _L)
    M = cross_n_l / np.sqrt(cross_n_l[0] ** 2 + cross_n_l[1] ** 2 + cross_n_l[2] ** 2)
    L = np.cross(M, N)

    # print(L, M, N)

    return L, M, N


def hybrid_mva(event_date, probe, outside_interval: int = 10, inside_interval: int = 2):
    duration = 4
    start_time = event_date - timedelta(hours=duration / 2)
    imported_data = HeliosData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                               duration=duration, probe=probe)
    imported_data.data.dropna(inplace=True)
    B = get_b(imported_data, event_date)
    L, M, N = mva(B)
    B1, B2, v1, v2, density_1, density_2, T_par_1, T_perp_1, T_par_2, T_perp_2 = get_side_data(imported_data,
                                                                                               event_date,
                                                                                               outside_interval,
                                                                                               inside_interval)
    L, M, N = hybrid(L, B1, B2)
    return L, M, N


def change_b_and_v(B1: np.ndarray, B2: np.ndarray, v1: np.ndarray, v2: np.ndarray, L: np.ndarray, M: np.ndarray,
                   N: np.ndarray):
    B1_L, B1_M, B1_N = np.dot(L, B1), np.dot(M, B1), np.dot(N, B1)
    B2_L, B2_M, B2_N = np.dot(L, B2), np.dot(M, B2), np.dot(N, B2)
    v1_L, v1_M, v1_N = np.dot(L, v1), np.dot(M, v1), np.dot(N, v1)
    v2_L, v2_M, v2_N = np.dot(L, v2), np.dot(M, v2), np.dot(N, v2)
    # print('bl', B1_L, B2_L)
    # print('bm', B1_M, B2_M)
    # print('bn', B1_N, B2_N)

    B1_changed, B2_changed = np.array([B1_L, B1_M, B1_N]), np.array([B2_L, B2_M, B2_N])
    v1_changed, v2_changed = np.array([v1_L, v1_M, v1_N]), np.array([v2_L, v2_M, v2_N])

    return B1_changed, B2_changed, v1_changed, v2_changed


def walen_test(B1_L: float, B2_L: float, v1_L: float, v2_L: float, rho_1: np.float64, rho_2: np.float64,
               minimum_fraction: float = 0.9, maximum_fraction: float = 1.1) -> bool:
    mu_0 = 4e-7 * np.pi
    k = 1.38e-23
    proton_mass = 1.67e-27

    rho_1 = rho_1 * proton_mass / 1e-15  # density is in cm-3, we want in km-3
    rho_2 = rho_2 * proton_mass / 1e-15  # density is in cm-3, we want in km-3
    alpha_1 = 0  # rho_1 * k * (T_par_1 - T_perp_1) / 2
    alpha_2 = 0  # rho_2 * k * (T_par_2 - T_perp_2) / 2
    B1_part = B1_L * np.sqrt((1 - alpha_1) / (mu_0 * rho_1)) * 10e-10  # b is in nanoteslas
    B2_part = B2_L * np.sqrt((1 - alpha_2) / (mu_0 * rho_2)) * 10e-10  # b is in nanoteslas

    # make sure that's the correct equation
    theoretical_v2_plus = v1_L + (B2_part - B1_part)
    theoretical_v2_minus = v1_L - (B2_part - B1_part)

    # the true v2 must be close to the predicted one, we will take the ones with same sign for comparison
    # if they all have the same sign, we compare to both v_plus and v_minus
    if np.sign(v2_L) == np.sign(theoretical_v2_plus) and np.sign(v2_L) == np.sign(theoretical_v2_minus):
        theoretical_v2 = np.min([np.abs(theoretical_v2_minus), np.abs(theoretical_v2_plus)])
        if minimum_fraction * theoretical_v2 < np.abs(v2_L) < maximum_fraction * theoretical_v2:
            return True
    elif np.sign(v2_L) == np.sign(theoretical_v2_plus):
        if minimum_fraction * np.abs(theoretical_v2_plus) < np.abs(v2_L) < maximum_fraction * np.abs(
                theoretical_v2_plus):
            return True
    elif np.sign(v2_L) == np.sign(theoretical_v2_minus):
        if minimum_fraction * np.abs(theoretical_v2_minus) < np.abs(v2_L) < maximum_fraction * np.abs(
                theoretical_v2_minus):
            return True
    else:
        print('wrong result')
    # print('real', v2_L)
    # print('theory', theoretical_v2_plus)
    # print('theory', theoretical_v2_minus)
    return False


def b_l_biggest(B1_L: float, B2_L: float, B1_M: float, B2_M: float) -> bool:
    amplitude_change_L = np.abs(B1_L - B2_L)
    amplitude_change_M = np.abs(B1_M - B2_M)
    magnitude_change_L = B1_L ** 2 + B2_L ** 2
    magnitude_change_M = B1_M ** 2 + B2_M ** 2

    if amplitude_change_L > 1 + amplitude_change_M or magnitude_change_L > 1 + magnitude_change_M:
        return True  # we do not want too close results, in which case it is not a reconnection
    else:
        return False


def changes_in_b_and_v(B1: np.ndarray, B2: np.ndarray, v1: np.ndarray, v2: np.ndarray, imported_data: ImportedData,
                       event_date: datetime, L: np.ndarray) -> bool:
    reconnection_points = 0

    # BL changes sign before and after the exhaust
    B1_L, B2_L = B1[0], B2[0]
    if np.sign(B1_L) != np.sign(B2_L):
        reconnection_points = reconnection_points + 1
    else:
        print('sign error')

    # bn is small and nearly constant
    B1_N, B2_N = B1[2], B2[2]
    if np.abs(B1_N) < 10e-15 and np.abs(B2_N) < 10e-15:
        reconnection_points = reconnection_points + 1
    else:
        print('bn too big')

    # changes in vm and vn are small compared to changes in vl
    delta_v = np.abs(v1 - v2)
    if delta_v[0] > delta_v[1] and delta_v[0] > delta_v[2]:
        reconnection_points = reconnection_points + 1
    else:
        print('v wrong')

    # changes in bl and vl are correlated on one side and anti-correlated on the other side
    BL, vL = [], []
    for n in range(len(imported_data.data)):
        BL.append(np.dot(
            np.array([imported_data.data['Bx'][n], imported_data.data['By'][n], imported_data.data['Bz'][n]]), L))
        vL.append(np.dot(
            np.array([imported_data.data['vp_x'][n], imported_data.data['vp_y'][n], imported_data.data['vp_z'][n]]), L))
    BL = pd.Series(np.array(BL), index=imported_data.data.index)
    vL = pd.Series(np.array(vL), index=imported_data.data.index)
    BL_diff = get_derivative(BL)
    vL_diff = get_derivative(vL)

    left_correlation = BL_diff.loc[
                       event_date - timedelta(minutes=15): event_date - timedelta(minutes=2)].values * vL_diff.loc[
                                                                                                       event_date - timedelta(
                                                                                                           minutes=15): event_date - timedelta(
                                                                                                           minutes=2)].values
    right_correlation = BL_diff.loc[
                        event_date + timedelta(minutes=2):event_date + timedelta(minutes=15)].values * vL_diff.loc[
                                                                                                       event_date + timedelta(
                                                                                                           minutes=2):event_date + timedelta(
                                                                                                           minutes=15)].values

    if np.sign(np.mean(left_correlation)) != np.sign(np.mean(right_correlation)):
        reconnection_points = reconnection_points + 1
    else:
        print('correlation error')

    if reconnection_points > 1:
        return True
    else:
        return False


def get_event_dates(file_name: str):
    event_dates = []
    with open(file_name) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year, month, day = np.int(row['year']), np.int(row['month']), np.int(row['day'])
            hours, minutes, seconds = np.int(row['hours']), np.int(row['minutes']), np.int(row['seconds'])
            event_dates.append(datetime(year, month, day, hours, minutes, seconds))
    return event_dates


def send_reconnections_to_csv(event_list: List[datetime], possible_reconnections_list: List[datetime], probe: int,
                              name: str = 'reconnections_tests'):
    with open(name + '.csv', 'w', newline='') as csv_file:
        fieldnames = ['year', 'month', 'day', 'hours', 'minutes', 'seconds', 'radius', 'satisfied tests']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for reconnection_date in event_list:
            year = reconnection_date.year
            month = reconnection_date.month
            day = reconnection_date.day
            hour = reconnection_date.hour
            minutes = reconnection_date.minute
            seconds = reconnection_date.second
            start = reconnection_date - timedelta(hours=1)
            imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=2,
                                         probe=probe)
            radius = imported_data.data['r_sun'].loc[
                     reconnection_date - timedelta(minutes=1): reconnection_date + timedelta(minutes=1)][0]
            if reconnection_date in possible_reconnections_list:
                satisfied = True
            else:
                satisfied = False
            writer.writerow(
                {'year': year, 'month': month, 'day': day, 'hours': hour, 'minutes': minutes, 'seconds': seconds,
                 'radius': radius, 'satisfied tests': satisfied})


def plot_all(imported_data: ImportedData, L: np.ndarray, M: np.ndarray, N: np.ndarray, event_date: datetime):

    bl, bm, bn = [], [], []
    vl, vm, vn = [], [], []
    for n in range(len(imported_data.data)):
        b = np.array([imported_data.data['Bx'][n], imported_data.data['By'][n], imported_data.data['Bz'][n]])
        # print(imported_data.data.index[n], b)
        v = np.array([imported_data.data['vp_x'][n], imported_data.data['vp_y'][n], imported_data.data['vp_z'][n]])
        bl.append(np.dot(b, L))
        bm.append(np.dot(b, M))
        bn.append(np.dot(b, N))
        vl.append(np.dot(v, L))
        vm.append(np.dot(v, M))
        vn.append(np.dot(v, N))

    bl = pd.Series(np.array(bl), index=imported_data.data.index)
    bm = pd.Series(np.array(bm), index=imported_data.data.index)
    bn = pd.Series(np.array(bn), index=imported_data.data.index)
    vl = pd.Series(np.array(vl), index=imported_data.data.index)
    vm = pd.Series(np.array(vm), index=imported_data.data.index)
    vn = pd.Series(np.array(vn), index=imported_data.data.index)

    imported_data.data['Bl'] = bl
    imported_data.data['Bm'] = bm
    imported_data.data['Bn'] = bn
    imported_data.data['v_l'] = vl
    imported_data.data['v_m'] = vm
    imported_data.data['v_n'] = vn

    plot_imported_data(imported_data, DEFAULT_PLOTTED_COLUMNS + [('Bl', 'v_l'), ('Bm', 'v_m'), ('Bn', 'v_n')],
                       save=False, event_date=event_date)


def test_reconnection_lmn(event_dates: List[datetime], probe: int, minimum_fraction: float, maximum_fraction: float,
                          plot: bool = False, mode: str = 'static'):
    """
    Checks a list of datetimes to determine whether they are reconnections
    :param event_dates: list of possible reconnections dates
    :param probe: probe to be analysed
    :param minimum_fraction: minimum walen fraction
    :param maximum_fraction: maximum walen fraction
    :param plot: bool, true of we want to plot reconnections that passed the test
    :param mode: intercative (human input to the code, more precise but time consuming) or static (purely computational)
    :return:
    """
    implemented_modes = ['static', 'interactive']
    if mode not in implemented_modes:
        raise NotImplementedError('This mode is not implemented.')
    duration = 4
    events_that_passed_test = []
    if mode == 'interactive':
        rogue_events = []
    for event_date in event_dates:
        try:
            start_time = event_date - timedelta(hours=duration / 2)
            imported_data = HeliosData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                                       duration=duration, probe=probe)
            imported_data.data.dropna(inplace=True)
            B = get_b(imported_data, event_date, 30)
            L, M, N = mva(B)
            B1, B2, v1, v2, density_1, density_2, T_par_1, T_perp_1, T_par_2, T_perp_2 = get_side_data(imported_data,
                                                                                                       event_date, 5, 1)
            L, M, N = hybrid(L, B1, B2)

            print('LMN:', L, M, N)

            B1_changed, B2_changed, v1_changed, v2_changed = change_b_and_v(B1, B2, v1, v2, L, M, N)
            B1_L, B2_L, B1_M, B2_M = B1_changed[0], B2_changed[0], B1_changed[1], B2_changed[1]
            v1_L, v2_L = v1_changed[0], v2_changed[0]

            walen = walen_test(B1_L, B2_L, v1_L, v2_L, density_1, density_2, minimum_fraction, maximum_fraction)
            BL_check = b_l_biggest(B1_L, B2_L, B1_M, B2_M)
            B_and_v_checks = changes_in_b_and_v(B1_changed, B2_changed, v1_changed, v2_changed, imported_data,
                                                event_date, L)
            if walen and BL_check and len(imported_data.data) > 70 and B_and_v_checks:  # avoid not enough data points
                print('RECONNECTION AT ', str(event_date))
                if mode == 'static':
                    events_that_passed_test.append(event_date)
                elif mode == 'interactive':
                    answered = False
                    # plot_all(imported_data, L, M, N, event_date)
                    while not answered:
                        is_event = str(input('Do you think this is an event?'))
                        is_event.lower()
                        if is_event[0] == 'y':
                            answered = True
                            events_that_passed_test.append(event_date)
                        elif is_event[0] == 'n':
                            answered = True
                            rogue_events.append(event_date)
                        else:
                            print('Please reply by yes or no')

                if plot and mode == 'static':
                    plot_all(imported_data, L, M, N, event_date)

            else:
                print('NO RECONNECTION AT ', str(event_date))
        except Exception:
            print('could not recover the data')

    if mode == 'interactive':
        print('rogue events: ', rogue_events)

    return events_that_passed_test


def test_reconnections_from_csv(file: str = 'reconnectionshelios2testdata1.csv', probe: int = 2, to_csv: bool = False,
                                plot: bool = True, mode='static'):
    event_dates = get_event_dates(file)
    probe = probe
    min_walen, max_walen = 0.9, 1.2
    events_that_passed_test = test_reconnection_lmn(event_dates, probe, min_walen, max_walen, plot=plot, mode=mode)
    print('number of reconnections: ', len(events_that_passed_test))
    if to_csv:
        send_reconnections_to_csv(event_dates, events_that_passed_test, probe=probe, name='reconnections_tests')


if __name__ == '__main__':
    # test_reconnections_from_csv('reconnections_helios_2_no_nt_27_19_5.csv', 2, plot=True, mode='interactive')
    # test_reconnection_lmn([datetime(1976, 12, 1, 6, 23)], 1, 0.9, 1.1, plot=True)
    test_reconnection_lmn([datetime(1978, 3, 3, 10, 56)], 1, 0.9, 1.1, plot=True)
