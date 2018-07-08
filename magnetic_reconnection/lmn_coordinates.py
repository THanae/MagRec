import numpy as np
from datetime import datetime, timedelta
from numpy import linalg as LA
import csv

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
    v_x_1 = np.mean(data_1['vp_x'].values)
    v_y_1 = np.mean(data_1['vp_y'].values)
    v_z_1 = np.mean(data_1['vp_z'].values)
    v1 = np.array([v_x_1, v_y_1, v_z_1])
    v_x_2 = np.mean(data_2['vp_x'].values)
    v_y_2 = np.mean(data_2['vp_y'].values)
    v_z_2 = np.mean(data_2['vp_z'].values)
    v2 = np.array([v_x_2, v_y_2, v_z_2])

    density_1 = np.mean(data_1['n_p'].values)
    density_2 = np.mean(data_2['n_p'].values)

    return B1, B2, v1, v2, density_1, density_2


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
    w_intermediate = np.min(np.delete([0, 1, 2], [w_min, w_max]))

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
    cross_of_b = np.cross(B1, B2)
    # we want a normalised vector
    N = cross_of_b/np.sqrt(cross_of_b[0]**2+cross_of_b[1]**2+cross_of_b[2]**2)
    cross_n_l = np.cross(N, _L)
    M = cross_n_l/np.sqrt(cross_n_l[0]**2 + cross_n_l[1]**2 + cross_n_l[2]**2)
    L = np.cross(M, N)
    print('L', L)
    print('M', M)
    print('N', N)
    return L, M, N


def change_b_and_v(B1, B2, v1, v2, L, M, N):
    B1_L = np.dot(L, B1)
    B1_M = np.dot(M, B1)
    B1_N = np.dot(N, B1)
    B2_L = np.dot(L, B2)
    B2_M = np.dot(M, B2)
    B2_N = np.dot(N, B2)
    v1_L = np.dot(L, v1)
    v1_M = np.dot(M, v1)
    v1_N = np.dot(N, v1)
    v2_L = np.dot(L, v2)
    v2_M = np.dot(M, v2)
    v2_N = np.dot(N, v2)

    # B1_changed = np.array([B1_L, B1_M, B1_N])
    # B2_changed = np.array([B2_L, B2_M, B2_N])
    # v1_changed = np.array([v1_L, v1_M, v1_N])
    # v2_changed = np.array([v2_L, v2_M, v2_N])

    return B1_L, B2_L, v1_L, v2_L


def walen_test(B1_L, B2_L, v1_L, v2_L, rho_1, rho_2):
    mu_0 = 4e-7 * np.pi
    alpha_1 = 0
    alpha_2 = 0
    B1_part = B1_L * np.sqrt((1-alpha_1)/(mu_0*rho_1))
    B2_part = B2_L * np.sqrt((1-alpha_2)/(mu_0*rho_2))
    theoretical_v2_plus = v1_L + (B2_part - B1_part)
    print('real', v2_L)
    print('theory', theoretical_v2_plus)
    theoretical_v2_minus = v1_L - (B2_part - B1_part)
    print('theory', theoretical_v2_minus)
    # the true v2 must be close to the predicted one
    if np.sign(v2_L) == np.sign(theoretical_v2_plus):
        if np.abs(v2_L) < np.abs(theoretical_v2_plus) < 10*np.abs(v2_L):
            print('reconnection')
        else:
            print('no reconnection')
    elif np.sign(v2_L) == np.sign(theoretical_v2_minus):
        if np.abs(v2_L) < np.abs(theoretical_v2_minus) < 10*np.abs(v2_L):
            print('reconnection')
        else:
            print('no reconnection')
    else:
        print('wrong result')


def get_event_dates(file_name):
    event_dates = []
    with open(file_name) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            year = np.int(row['year'])
            month = np.int(row['month'])
            day = np.int(row['day'])
            hours = np.int(row['hours'])
            minutes = np.int(row['minutes'])
            event_dates.append(datetime(year, month, day, hours, minutes))
    return event_dates


if __name__ == '__main__':
    event_dates = get_event_dates('reconnections_all_of_them.csv')
    duration = 1.5
    for event_date in event_dates:
        print('possible reconnection', str(event_date))
        start_time = event_date - timedelta(hours=duration/2)
        imported_data = ImportedData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour, duration=duration, probe=2)
        imported_data.data.dropna(inplace=True)
        # print(imported_data.data.keys())
        # print(imported_data.data)
        B = get_b(imported_data)
        L = mva(B)
        B1, B2, v1, v2, density_1, density_2 = get_necessary_b(imported_data, event_date)
        L, M, N = hybrid(L, B1, B2)
        B1_L, B2_L, v1_L, v2_L = change_b_and_v(B1, B2, v1, v2, L, M, N)
        walen_test(B1_L, B2_L, v1_L, v2_L, density_1, density_2)
