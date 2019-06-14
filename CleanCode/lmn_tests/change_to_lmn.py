from datetime import timedelta, datetime
from typing import List, Tuple
import numpy as np
from numpy import linalg as la
import pandas as pd


def get_side_data_v_and_b(imported_data, event_date: datetime, outside_interval: int = 10,
                          inside_interval: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get side data for the velocity and magnetic field
    :param imported_data: data to analyse
    :param event_date: date of potential event
    :param outside_interval: outside limit of the data to be considered (consider from e. date - o_i to e. date - i.i)
    :param inside_interval: inside limit of the data to be considered
    :return:
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

    return b1, b2, v1, v2


def mva(b: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MVA analysis of the magnetic field in order to get LMN coordinates (part 1 of hybrid MVA)
    :param b: magnetic field from which to find LMN coordinates
    :return:
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
        L[coordinate]= v[:, w_max][coordinate]
        N[coordinate] = v[:, w_min][coordinate]
        M[coordinate] = v[:, w_intermediate][coordinate]

    return L, M, N


def hybrid(_l: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use MVA analysis and side data to find more accurate version of LMN coordinates
    :param _l: intermediate L
    :param b1: b on the left hand side of the data
    :param b2: b on the right hand side of the data
    :return:
    """
    cross_of_b = np.cross(b1, b2)
    N = cross_of_b / np.sqrt(cross_of_b[0] ** 2 + cross_of_b[1] ** 2 + cross_of_b[2] ** 2)  # normalised vector
    cross_n_l = np.cross(N, _l)
    M = cross_n_l / np.sqrt(cross_n_l[0] ** 2 + cross_n_l[1] ** 2 + cross_n_l[2] ** 2)
    L = np.cross(M, N)
    return L, M, N


def hybrid_mva(imported_data, event_date, outside_interval: int = 10, inside_interval: int = 2,
               mva_interval: int = 30)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds LMN with hybrid mva
    :param imported_data: data to perform the analysis on
    :param event_date: date of the potential event
    :param outside_interval: most outward limit to consider for the side data
    :param inside_interval: inward limit to consider for the side data
    :param mva_interval: interval on each side of the event to consider for the MVA analysis
    :return:
    """
    imported_data.data.dropna(inplace=True)
    b = []
    data = imported_data.data[event_date - timedelta(minutes=mva_interval):event_date + timedelta(minutes=mva_interval)]
    b_x, b_y, b_z = data['Bx'].values, data['By'].values, data['Bz'].values
    for n in range(len(b_x)):
        b.append(np.array([b_x[n], b_y[n], b_z[n]]))
    L, M, N = mva(b)
    b1, b2, v1, v2 = get_side_data_v_and_b(imported_data, event_date, outside_interval, inside_interval)
    L, M, N = hybrid(L, b1, b2)
    return L, M, N


def change_coordinates_to_lmn(imported_data, L, M=None, N=None) -> None:
    """
    Creates new columns for the magnetic field and the velocity in LMN coordinates
    :param imported_data: data to perform analysis on
    :param L: L vector
    :param M: M vector
    :param N: N vector
    :return:
    """
    bl, bm, bn = [], [], []
    vl, vm, vn = [], [], []
    for n in range(len(imported_data.data)):
        b = np.array([imported_data.data['Bx'][n], imported_data.data['By'][n], imported_data.data['Bz'][n]])
        v = np.array([imported_data.data['vp_x'][n], imported_data.data['vp_y'][n], imported_data.data['vp_z'][n]])
        if M is None and N is None:  # sometimes just L is required, then m and N are not needed
            M, N = [0, 0, 0], [0, 0, 0]
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
