from datetime import timedelta, datetime
from typing import List, Tuple
import numpy as np
from numpy import linalg as la

from data_handler.data_importer.data_import import get_probe_data
from data_handler.data_importer.imported_data import ImportedData


def get_b(imported_data: ImportedData, event_date, interval: int = 30) -> List[np.ndarray]:
    """
    Returns the imported data in a suitable vector form to be analysed
    :param imported_data: ImportedData
    :param event_date: time of the possible reconnection event
    :param interval: interval over which we get the magnetic field
    :return: array [bx, by, bz]
    """
    b = []
    data = imported_data.data[event_date - timedelta(minutes=interval):event_date + timedelta(minutes=interval)]
    b_x, b_y, b_z = data['Bx'].values, data['By'].values, data['Bz'].values
    for n in range(len(b_x)):
        b.append(np.array([b_x[n], b_y[n], b_z[n]]))
    return b


def get_side_data(imported_data: ImportedData, event_date: datetime, outside_interval: int = 10,
                  inside_interval: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns B around the reconnection event. We need B to be stable over a period of time on both sides of the exhaust.
    We then take the average in that region. Hard to determine computationally, so use 10-2 intervals
    :param imported_data: ImportedData
    :param event_date: date of possible reconnection
    :param outside_interval: interval outside of the event that will be considered
    :param inside_interval: interval inside the event that will be considered
    :return: B around the reconnection event
    """
    data_1 = imported_data.data[
             event_date - timedelta(minutes=outside_interval):event_date - timedelta(minutes=inside_interval)]
    data_2 = imported_data.data[
             event_date + timedelta(minutes=inside_interval):event_date + timedelta(minutes=outside_interval)]

    b_x_1, b_y_1, b_z_1 = np.mean(data_1['Bx'].values), np.mean(data_1['By'].values), np.mean(data_1['Bz'].values)
    b_x_2, b_y_2, b_z_2 = np.mean(data_2['Bx'].values), np.mean(data_2['By'].values), np.mean(data_2['Bz'].values)
    b1, b2 = np.array([b_x_1, b_y_1, b_z_1]), np.array([b_x_2, b_y_2, b_z_2])

    v_x_1, v_y_1, v_z_1 = np.mean(data_1['vp_x'].values), np.mean(data_1['vp_y'].values), np.mean(data_1['vp_z'].values)
    v_x_2, v_y_2, v_z_2 = np.mean(data_2['vp_x'].values), np.mean(data_2['vp_y'].values), np.mean(data_2['vp_z'].values)
    v1, v2 = np.array([v_x_1, v_y_1, v_z_1]), np.array([v_x_2, v_y_2, v_z_2])

    density_1, density_2 = np.mean(data_1['n_p'].values), np.mean(data_2['n_p'].values)
    if imported_data.probe == 'wind':
        t_par_1, t_perp_1 = np.array([0]), np.array([0])
        t_par_2, t_perp_2 = np.array([0]), np.array([0])
    else:
        t_par_1, t_perp_1 = np.mean(data_1['Tp_par'].values), np.mean(data_1['Tp_perp'].values)
        t_par_2, t_perp_2 = np.mean(data_2['Tp_par'].values), np.mean(data_2['Tp_perp'].values)

    return b1, b2, v1, v2, density_1, density_2, t_par_1, t_perp_1, t_par_2, t_perp_2


def mva(b: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the LMN component of the new coordinates system, by solving the magnetic matrix eigenvalue problem
    :param b: field around interval that will be considered
    :return: the L, M, and N vectors
    """
    magnetic_matrix = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for n in range(3):
        bn = np.array([b[n] for b in b])
        for m in range(3):
            bm = np.array([b[m] for b in b])

            magnetic_matrix[n, m] = np.mean(bn * bm) - np.mean(bn) * np.mean(bm)

    w, v = la.eig(magnetic_matrix)
    w_max = np.argmax(w)  # maximum value gives L
    w_min = np.argmin(w)  # minimum eigenvalue gives N
    w_intermediate = np.min(np.delete([0, 1, 2], [w_min, w_max]))

    L, N, M = np.zeros(3), np.zeros(3), np.zeros(3)
    for coordinate in range(len(v[:, w_max])):
        L[coordinate] = v[:, w_max][coordinate]
        N[coordinate] = v[:, w_min][coordinate]
        M[coordinate] = v[:, w_intermediate][coordinate]

    return L, M, N


def hybrid(_l: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the other components of the new coordinates system, useful if eigenvalues not well resolved
    :param _l: L vector found with mva
    :param b1: mean magnetic field vector from inflow region 1
    :param b2: mean magnetic field vector from inflow region 2
    :return: L, M, N vectors
    """
    cross_of_b = np.cross(b1, b2)
    N = cross_of_b / np.sqrt(cross_of_b[0] ** 2 + cross_of_b[1] ** 2 + cross_of_b[2] ** 2)  # normalised vector
    cross_n_l = np.cross(N, _l)
    M = cross_n_l / np.sqrt(cross_n_l[0] ** 2 + cross_n_l[1] ** 2 + cross_n_l[2] ** 2)
    L = np.cross(M, N)
    return L, M, N


def hybrid_mva(event_date, probe, duration: int = 4, outside_interval: int = 10, inside_interval: int = 2,
               mva_interval: int = 30) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_time = event_date - timedelta(hours=duration / 2)
    imported_data = get_probe_data(probe=probe, start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour,
                                   duration=duration)
    imported_data.data.dropna(inplace=True)
    b = get_b(imported_data, event_date, interval=mva_interval)
    L, M, N = mva(b)
    b1, b2, v1, v2, density_1, density_2, t_par_1, t_perp_1, t_par_2, t_perp_2 = get_side_data(imported_data,
                                                                                               event_date,
                                                                                               outside_interval,
                                                                                               inside_interval)
    L, M, N = hybrid(L, b1, b2)
    return L, M, N
