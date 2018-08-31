import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas as pd
from typing import List, Tuple, Union, Optional

from data_handler.data_importer.data_import import get_probe_data
from data_handler.data_importer.imported_data import ImportedData
from data_handler.imported_data_plotter import plot_imported_data, DEFAULT_PLOTTED_COLUMNS
from data_handler.utils.column_processing import get_derivative
from magnetic_reconnection_dir.csv_utils import get_dates_from_csv, send_dates_to_csv
from magnetic_reconnection_dir.mva_analysis import get_b, mva, hybrid, get_side_data

mu_0 = 4e-7 * np.pi
k = 1.38e-23
proton_mass = 1.67e-27


def change_b_and_v(B1: np.ndarray, B2: np.ndarray, v1: np.ndarray, v2: np.ndarray, L: np.ndarray, M: np.ndarray,
                   N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds B and v in LMN coordinates
    :param B1: B on the left of the event
    :param B2: B on the right of the event
    :param v1: v on the left of the event
    :param v2: v on the right of the event
    :param L: L vector
    :param M: M vector
    :param N: N vector
    :return: the changed B and v
    """
    B1_L, B1_M, B1_N = np.dot(L, B1), np.dot(M, B1), np.dot(N, B1)
    B2_L, B2_M, B2_N = np.dot(L, B2), np.dot(M, B2), np.dot(N, B2)
    v1_L, v1_M, v1_N = np.dot(L, v1), np.dot(M, v1), np.dot(N, v1)
    v2_L, v2_M, v2_N = np.dot(L, v2), np.dot(M, v2), np.dot(N, v2)
    B1_changed, B2_changed = np.array([B1_L, B1_M, B1_N]), np.array([B2_L, B2_M, B2_N])
    v1_changed, v2_changed = np.array([v1_L, v1_M, v1_N]), np.array([v2_L, v2_M, v2_N])
    return B1_changed, B2_changed, v1_changed, v2_changed


def get_alfven_speed(B1_L: float, B2_L: float, v1_L: float, v2_L: float, rho_1: np.float64, rho_2: np.float64) -> Tuple[
                     float, float]:
    """
    Finds the Alfven speed at the boundaries of the exhaust
    :param B1_L: left B_L
    :param B2_L: right B_L
    :param v1_L: left v_L
    :param v2_L: right v_l
    :param rho_1: left density
    :param rho_2: right density
    :return: the Alfven speed
    """
    rho_1 = rho_1 * proton_mass / 1e-15  # density is in cm-3, we want in km-3
    rho_2 = rho_2 * proton_mass / 1e-15  # density is in cm-3, we want in km-3
    alpha_1 = 0  # rho_1 * k * (T_par_1 - T_perp_1) / 2
    alpha_2 = 0  # rho_2 * k * (T_par_2 - T_perp_2) / 2
    B1_part = B1_L * np.sqrt((1 - alpha_1) / (mu_0 * rho_1)) * 10e-10  # b is in nT
    B2_part = B2_L * np.sqrt((1 - alpha_2) / (mu_0 * rho_2)) * 10e-10  # b is in nT

    theoretical_v2_plus = v1_L + (B2_part - B1_part)
    theoretical_v2_minus = v1_L - (B2_part - B1_part)

    return theoretical_v2_plus, theoretical_v2_minus


def walen_test(B1_L: float, B2_L: float, v1_L: float, v2_L: float, rho_1: np.float64, rho_2: np.float64,
               minimum_fraction: float = 0.9, maximum_fraction: float = 1.1) -> bool:
    """
    Applied the Walen test at the boundaries of the exhaust
    :param B1_L: left B_L
    :param B2_L: right B_L
    :param v1_L: left v_L
    :param v2_L: right v_l
    :param rho_1: left density
    :param rho_2: right density
    :param minimum_fraction: minimum fraction of the Alfven speed that the speed must be to satisfy the walen test
    :param maximum_fraction: maximum fraction of the Alfven speed that the speed must be to satisfy the walen test
    :return: True if satisfied, False if not satisfied
    """
    theoretical_v2_plus, theoretical_v2_minus = get_alfven_speed(B1_L, B2_L, v1_L, v2_L, rho_1, rho_2)

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
    return False


def plot_walen_test(event_date: datetime, probe: int, duration: int = 4, outside_interval: int = 10,
                    inside_interval: int = 2) ->List[List[Union[str, datetime, float]]]:
    """
    Plots the expected Alfven speed at the boundaries of the exhaust (work in progress)
    :param event_date: date of the reconnection event
    :param probe: probe that detected the event
    :param duration: duration that we download the data for
    :param outside_interval: interval outside the event that we consider the data for
    :param inside_interval: interval inside the event that we consider the data for
    :return: Alfven speed at the exhaust
    """
    start_time = event_date - timedelta(hours=duration / 2)
    imported_data = get_probe_data(probe=probe, start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                                   duration=duration)
    imported_data.data.dropna(inplace=True)
    B = get_b(imported_data, event_date, 30)
    L, M, N = mva(B)
    B1, B2, v1, v2, density_1, density_2, T_par_1, T_perp_1, T_par_2, T_perp_2 = get_side_data(imported_data,
                                                                                               event_date,
                                                                                               outside_interval,
                                                                                               inside_interval)
    L, M, N = hybrid(L, B1, B2)
    print('LMN:', L, M, N)

    B1_changed, B2_changed, v1_changed, v2_changed = change_b_and_v(B1, B2, v1, v2, L, M, N)
    B1_L, B2_L, B1_M, B2_M = B1_changed[0], B2_changed[0], B1_changed[1], B2_changed[1]
    v1_L, v2_L = v1_changed[0], v2_changed[0]
    theoretical_v2_plus, theoretical_v2_minus = get_alfven_speed(B1_L, B2_L, v1_L, v2_L, density_1, density_2)
    theoretical_time = event_date + timedelta(minutes=inside_interval / 2)

    return [['v_l', theoretical_time, theoretical_v2_plus], ['v_l', theoretical_time, theoretical_v2_minus]]


def b_l_biggest(B1_L: float, B2_L: float, B1_M: float, B2_M: float) -> bool:
    """
    Checks whether Bl has a bigger magnitude change than Bm
    :param B1_L: left B_L
    :param B2_L: right B_L
    :param B1_M: left B_M
    :param B2_M: right B_M
    :return: True if Bl is bigger, False otherwise
    """
    amplitude_change_l, amplitude_change_m = np.abs(B1_L - B2_L), np.abs(B1_M - B2_M)
    magnitude_change_l, magnitude_change_m = B1_L ** 2 + B2_L ** 2, B1_M ** 2 + B2_M ** 2
    if amplitude_change_l > 1 + amplitude_change_m or magnitude_change_l > 1 + magnitude_change_m:
        return True  # we do not want too close results, in which case it is not a reconnection
    else:
        return False


def changes_in_b_and_v(B1: np.ndarray, B2: np.ndarray, v1: np.ndarray, v2: np.ndarray, imported_data: ImportedData,
                       event_date: datetime, L: np.ndarray) -> bool:
    """
    Checks whether the magnetic field and velocity satisfy common reconnection events features
    :param B1: left changed B
    :param B2: right changed B
    :param v1: left changed v
    :param v2: right changed v
    :param imported_data: ImportedData
    :param event_date: date and time of the reconnection event
    :param L: L vector
    :return: True if at least two tests are satisfied, False otherwise
    """
    reconnection_points = 0

    # BL changes sign before and after the exhaust
    B1_L, B2_L = B1[0], B2[0]
    if np.sign(B1_L) != np.sign(B2_L):
        reconnection_points = reconnection_points + 1
    else:
        print('sign error')
        return False

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
                       event_date - timedelta(minutes=15): event_date - timedelta(minutes=2)].values
    right_correlation = BL_diff.loc[
                        event_date + timedelta(minutes=2):event_date + timedelta(minutes=15)].values * vL_diff.loc[
                        event_date + timedelta(minutes=2):event_date + timedelta(minutes=15)].values

    if np.sign(np.mean(left_correlation)) != np.sign(np.mean(right_correlation)):
        reconnection_points = reconnection_points + 1
    else:
        print('correlation error')
    if reconnection_points > 1:
        return True
    else:
        return False


def plot_lmn(imported_data: ImportedData, L: np.ndarray, M: np.ndarray, N: np.ndarray, event_date: datetime, probe: int,
             boundaries: Optional[List[datetime]]=None):
    """
    Plots the magnetic field and velocity in LMN coordinates
    :param imported_data: ImportedData
    :param L: L vector
    :param M: M vector
    :param N: N vector
    :param event_date: date and time of the reconnection event
    :param probe: probe that detected the event
    :param boundaries: boundaries to be plotted, defaults to None
    :return:
    """
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

    imported_data.data['Bl'], imported_data.data['Bm'], imported_data.data['Bn'] = bl, bm, bn
    imported_data.data['v_l'], imported_data.data['v_m'], imported_data.data['v_n'] = vl, vm, vn

    # scatter_points = plot_walen_test(event_date=event_date, probe=probe)
    plot_imported_data(imported_data, DEFAULT_PLOTTED_COLUMNS + [('Bl', 'v_l'), ('Bm', 'v_m'), ('Bn', 'v_n')],
                       save=False, event_date=event_date, boundaries=boundaries)


def test_reconnection_lmn(event_dates: List[datetime], probe: int, minimum_fraction: float, maximum_fraction: float,
                          plot: bool = False, mode: str = 'static') ->List[datetime]:
    """
    Checks a list of type datetime to determine whether they are reconnection events
    :param event_dates: list of possible reconnection dates
    :param probe: probe to be analysed
    :param minimum_fraction: minimum walen fraction
    :param maximum_fraction: maximum walen fraction
    :param plot: bool, true of we want to plot reconnection events that passed the test
    :param mode: interactive (human input to the code, more precise but time consuming) or static (purely computational)
    :return: all events that managed to pass the lmn tests
    """
    implemented_modes = ['static', 'interactive']
    if mode not in implemented_modes:
        raise NotImplementedError('This mode is not implemented.')
    duration = 4
    events_that_passed_test = []
    known_events = get_dates_from_csv('helios2_magrec2.csv')
    rogue_events = []  # if mode == 'interactive'
    for event_date in event_dates:
        # try:
        start_time = event_date - timedelta(hours=duration / 2)
        imported_data = get_probe_data(probe=probe, start_date=start_time.strftime('%d/%m/%Y'),
                                       start_hour=start_time.hour, duration=duration)
        imported_data.data.dropna(inplace=True)
        if probe == 1 or probe == 2:
            B = get_b(imported_data, event_date, 30)
            L, M, N = mva(B)
            B1, B2, v1, v2, density_1, density_2, T_par_1, T_perp_1, T_par_2, T_perp_2 = get_side_data(imported_data,
                                                                                                       event_date, 10,
                                                                                                       2)
            min_len = 70
        elif probe == 'ulysses':
            B = get_b(imported_data, event_date, 60)
            L, M, N = mva(B)
            B1, B2, v1, v2, density_1, density_2, T_par_1, T_perp_1, T_par_2, T_perp_2 = get_side_data(imported_data,
                                                                                                       event_date, 30,
                                                                                                       10)
            min_len = 5
        else:
            raise NotImplementedError('The probes that have been implemented so far are Helios 1, Helios 2 and Ulysses')
        L, M, N = hybrid(L, B1, B2)
        print('LMN:', L, M, N)
        print(np.dot(L, M), np.dot(L, N), np.dot(M, N), np.dot(np.cross(L, M), N))

        B1_changed, B2_changed, v1_changed, v2_changed = change_b_and_v(B1, B2, v1, v2, L, M, N)
        B1_L, B2_L, B1_M, B2_M = B1_changed[0], B2_changed[0], B1_changed[1], B2_changed[1]
        v1_L, v2_L = v1_changed[0], v2_changed[0]

        walen = walen_test(B1_L, B2_L, v1_L, v2_L, density_1, density_2, minimum_fraction, maximum_fraction)
        bl_check = b_l_biggest(B1_L, B2_L, B1_M, B2_M)
        b_and_v_checks = changes_in_b_and_v(B1_changed, B2_changed, v1_changed, v2_changed, imported_data,
                                            event_date, L)

        if walen and bl_check and len(imported_data.data) > min_len and b_and_v_checks:  # avoid not enough data points
            print('RECONNECTION ON ', str(event_date))
            if mode == 'static':
                events_that_passed_test.append(event_date)
            elif mode == 'interactive':
                answered = False
                plot_lmn(imported_data, L, M, N, event_date, probe, boundaries=None)
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

            if plot and mode == 'static' and event_date not in known_events:
                plot_lmn(imported_data, L, M, N, event_date, probe=probe)

        else:
            print('NO RECONNECTION ON ', str(event_date))
        # except Exception:
        #     print('could not recover the data')

    if mode == 'interactive':
        print('rogue events: ', rogue_events)

    return events_that_passed_test


def test_reconnection_events_from_csv(file: str = 'reconnectionshelios2testdata1.csv', probe: Union[int, str] = 2,
                                      to_csv: bool = False, plot: bool = False, mode: str ='static'):
    """
    Tests reconnection events from a csv file
    :param file: name of the file
    :param probe: probe that detected the events
    :param to_csv: if True, sends the found reconnection events to a csv file if they pass the tests
    :param plot: if True, plots the events that passed tests
    :param mode: defaults to static, if interactive, asks the person if the found event is an event
    :return:
    """
    event_dates = get_dates_from_csv(file)
    probe = probe
    min_walen, max_walen = 0.9, 1.2
    events_that_passed_test = test_reconnection_lmn(event_dates, probe, min_walen, max_walen, plot=plot, mode=mode)
    print('number of reconnection events: ', len(events_that_passed_test))
    if to_csv:
        filename = 'probe' + str(probe) + 'reconnection_events'
        send_dates_to_csv(filename=filename, events_list=events_that_passed_test, probe=probe)


if __name__ == '__main__':
    # test_reconnection_events_from_csv('helios2mag_rec3.csv', 2, plot=True, mode='static')
    # test_reconnection_events_from_csv('helios2_magrec.csv', 2, plot=True, mode='static')
    # test_reconnection_lmn([datetime(1976, 12, 1, 6, 23)], 1, 0.9, 1.1, plot=True)
    # test_reconnection_lmn([datetime(1978, 3, 3, 10, 56)], 1, 0.9, 1.1, plot=True)
    test_reconnection_events_from_csv('reconnections_helios_ulysses_no_nt_3_25_30.csv', probe='ulysses', plot=True)
    # test_reconnection_lmn([datetime(1977, 11, 25, 4, 47)], probe=1, minimum_fraction=0.9, maximum_fraction=1.1,
    #                       plot=True)
    # test_reconnection_lmn([datetime(1977, 11, 23, 9, 27)], probe=2, minimum_fraction=0.9, maximum_fraction=1.1,
    #                       plot=True)
    # # test_reconnection_lmn([datetime(1977, 2, 5, 18, 30)], probe=2, minimum_fraction=0.9, maximum_fraction=1.1,
    # #                       plot=True)
