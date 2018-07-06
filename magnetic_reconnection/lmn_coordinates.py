import numpy as np
from datetime import datetime, timedelta
from numpy import linalg as LA

from data_handler.imported_data import ImportedData

# test data
# B = [np.array([-13.6, -24.7, 54.6]),
#          np.array([-14.8, -24.9, 58.7]),
#          np.array([-13.4, -17.2, 62.4]),
#          np.array([-14.0, -25.0, 43.8]),
#          np.array([-7.1, -4.5, 33.5]),
#          np.array([-0.9, -5.0, 44.4]),
#          np.array([-10.0, -0.4, 44.6]),
#          np.array([-6.1, -4.8, 21.1]),
#          np.array([1.2, 1.6, 21.1]),
#          np.array([-3.4, -3.9, 4.1]),
#          np.array([-0.9, 1.2, 5.0]),
#          np.array([-1.0, -1.5, 12.3]),
#          np.array([11.0, 13.2, 29.7]),
#          np.array([19.1, 34.4, 20.1]),
#          np.array([24.9, 50.1, 1.9]),
#          np.array([29.2, 47.1, -10.6])]


def get_b(imported_data):
    """
    Returns the imported data in a suitable vector form to be analysed
    :param imported_data:
    :return:
    """
    B = []
    b_x = imported_data.data['Bx'].values
    b_y = imported_data.data['By'].values
    b_z = imported_data.data['Bz'].values
    for n in range(len(b_x)):
        B.append(np.array([b_x[n], b_y[n], b_z[n]]))
    return B


def get_necessary_b(imported_data, event_date):
    """
    Returns B1 and B2 around the reconnection event
    :param imported_data: ImportedData
    :param event_date: date of possible reconnection
    :return:
    """
    # now we want to make sure we take only the B we need
    # we need B to be stable over some period of time on both sides of the exhaust
    # we take the average in that stability region
    # we take off the reconnection event (few minutes on each side of the event)

    # time interval is 3-30 by default, but need to change it for turbulent data/lack of data
    data_1 = imported_data.data[event_date - timedelta(minutes=30):event_date - timedelta(minutes=3)]
    data_2 = imported_data.data[event_date+timedelta(minutes=3):event_date+timedelta(minutes=30)]
    b_x_1 = np.mean(data_1['Bx'].values)
    b_y_1 = np.mean(data_1['By'].values)
    b_z_1 = np.mean(data_1['Bz'].values)
    B1 = np.array([b_x_1, b_y_1, b_z_1])
    b_x_2 = np.mean(data_2['Bx'].values)
    b_y_2 = np.mean(data_2['By'].values)
    b_z_2 = np.mean(data_2['Bz'].values)
    B2 = np.array([b_x_2, b_y_2, b_z_2])
    v_x_1 = np.mean(data_1['v_x'].values)
    v_y_1 = np.mean(data_1['v_y'].values)
    v_z_1 = np.mean(data_1['v_z'].values)
    v1 = np.array([v_x_1, v_y_1, v_z_1])
    v_x_2 = np.mean(data_2['v_x'].values)
    v_y_2 = np.mean(data_2['v_y'].values)
    v_z_2 = np.mean(data_2['v_z'].values)
    v2 = np.array([v_x_2, v_y_2, v_z_2])


    return B1, B2, v1, v2


def mva(B):
    """
    Finds the L component of the new coordinates system
    :param B: field around interval that will be considered
    :return:
    """
    # we want to solve the matrix
    M_b = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for n in range(3):
        bn = np.array([b[n] for b in B])
        for m in range(3):
            bm = np.array([b[m] for b in B])

            M_b[n, m] = np.mean(bn*bm) - np.mean(bn)*np.mean(bm)

    w, v = LA.eig(M_b)
    # minimum eigenvalue gives N
    # maximum value gives L
    w_max = np.argmax(w)
    w_min = np.argmin(w)
    w_intermediate = np.delete([0, 1, 2], [w_min, w_max])

    L = np.zeros(3)
    for coordinate in range(len(v[:, w_max])):
        L[coordinate] = v[:, w_max][coordinate]

    N = np.zeros(3)
    for coordinate in range(len(v[:,  w_min])):
        N[coordinate] = v[:, w_min][coordinate]

    M = np.zeros(3)
    for coordinate in range(len(v[:, w_intermediate])):
        M[coordinate] = v[:, w_intermediate][coordinate]

    b_av = np.array([0,0,0])
    b_l = 0
    for n in range(3):
        b_av[n] = np.mean(np.array([b[n] for b in B]))

    for n in range(3):
        b_l = b_l + b_av[n]*L[n]
    return L


# hybrid mva necessary if eigenvalues not well resolved
def hybrid(_L, B1, B2):
    """
    Finds the other components of the new coordinates system
    :param _L: L vector found with mva
    :param B1: mean magnetic field vector from inflow region 1
    :param B2: mean magnetic field vector from inflow region 2
    :return:
    """
    # take average b on the left
    # take average b on the right
    # take only average for stable regions, so maybe no more deviations than one or two sigmas
    b_mean = np.array([np.mean(np.array([b[n] for b in B]))for n in range(3)])
    cross_of_b = np.cross(b_mean, b_mean-np.array([1,1,1]))
    # we want a normalised vector
    N = cross_of_b/np.sqrt(cross_of_b[0]**2+cross_of_b[1]**2+cross_of_b[2]**2)
    cross_n_l = np.cross(N, _L)
    M = cross_n_l/np.sqrt(cross_n_l[0]**2 + cross_n_l[1]**2 + cross_n_l[2]**2)
    L = np.cross(M, N)
    print('L', L)
    print('M', M)
    print('N', N)
    return L, M, N


if __name__ == '__main__':
    imported_data = ImportedData(start_date='30/01/1976', start_hour=1, duration=1.5)
    imported_data.data.dropna(inplace=True)
    print(imported_data.data.keys())
    # print(imported_data.data)
    B = get_b(imported_data)
    L = mva(B)
    B1, B2, v1, v2 = get_necessary_b(imported_data, datetime(1976, 1, 30, 1, 40, 0))
    L, M, N = hybrid(L, B1, B2)


# need another function that applies the walen test
