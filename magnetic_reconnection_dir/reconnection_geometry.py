from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D

from data_handler.data_importer.helios_data import HeliosData
from magnetic_reconnection_dir.lmn_coordinates import hybrid_mva


def compare_lmn(event_date: datetime, weird_thing_date: datetime, probe: int, outside_interval: int = 10,
                inside_interval: int = 2):
    """
    Gets the lmn coordinates for the event and the other event close to it
    :param event_date: time of the event
    :param weird_thing_date: time of the weird thing
    :param probe: 1 or 2 for Helios 1 or 2
    :return:
    """
    L, M, N = hybrid_mva(event_date, probe, outside_interval, inside_interval)
    L_w, M_w, N_w = hybrid_mva(weird_thing_date, probe, outside_interval, inside_interval)
    print('dot products', np.dot(L, M), np.dot(L, N), np.dot(M, N))
    print('dot products', np.dot(L_w, M_w), np.dot(L_w, N_w), np.dot(M_w, N_w))
    print('handedness tests', np.dot(np.cross(L, M), N), np.dot(np.cross(L_w, M_w), N_w))
    return [L, M, N], [L_w, M_w, N_w]


def get_rotation_matrix(event: List[np.ndarray], weird: List[np.ndarray]) -> np.ndarray:
    """
    Finds the rotation matrix between the two events
    :param event: LMN coordinates of the reconnection event
    :param weird: LMN coordinates of the other thing
    :return:
    """
    rotation = np.matmul(np.matrix(weird), inv(np.matrix(event)))  # rotation * event = weird
    print('det', np.linalg.det(rotation))
    return rotation


def rotation_matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
    """
    Finds the rotation angles between the two events coordinates
    :param rotation: rotation matrix
    :return:
    """
    if rotation[2, 0] < -0.9 or rotation[2, 0] > 0.9:
        singular = True
    else:
        singular = False
    if not singular:
        theta = np.arcsin(-rotation[2, 0])
        psi = np.arctan2(rotation[2, 1] * np.sign(np.cos(theta)), rotation[2, 2] * np.sign(np.cos(theta)))
        phi = np.arctan2(rotation[1, 0] * np.sign(rotation[2, 0]), rotation[0, 0] * np.sign(rotation[2, 0]))
    else:
        theta = np.arccos(-rotation[2, 0])
        psi = np.arctan2(rotation[2, 1] * np.sign(rotation[2, 0]), rotation[2, 2] * np.sign(rotation[2, 0]))
        phi = 0
    print('theta, psi, phi', np.degrees(np.array([theta, psi, phi])))
    return np.array([theta, psi, phi])


def plot_vectors(event: List[np.ndarray], weird: List[np.ndarray]):
    fig = plt.figure()
    origin = [0, 0, 0]
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    X, Y, Z = zip(origin, origin, origin)

    colors = ['#ff0000', '#000066', '#006600'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    for n in range(len(event)):
        U, V, W = zip(event[n])
        ax.quiver(X, Y, Z, U, V, W, color=colors[n])
        ax.quiver(X, Y, -1, np.dot(x, event[n]), np.dot(y, event[n]), 0, color='#808080')
        ax.quiver(X, 1, Z, np.dot(x, event[n]), 0, np.dot(z, event[n]), color='#808080')
        ax.quiver(-1, Y, Z, 0, np.dot(y, event[n]), np.dot(z, event[n]), color='#808080')

    colors = ['#ff6666', '#3366ff', '#00ff00'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    for n in range(len(weird)):
        U, V, W = zip(weird[n])
        ax.quiver(X, Y, Z, U, V, W, color=colors[n])
        ax.quiver(X, Y, -1, np.dot(x, weird[n]), np.dot(y, weird[n]), 0, color='#808080')
        ax.quiver(X, 1, Z, np.dot(x, weird[n]), 0, np.dot(z, weird[n]), color='#808080')
        ax.quiver(-1, Y, Z, 0, np.dot(y, weird[n]), np.dot(z, weird[n]), color='#808080')
    plt.show()


def plot_2d_3d(fig_name, lmn_coordinates: List[np.ndarray], colors: list):
    """
    Plots the 3d and 2d data
    :param fig_name: name of the figure to complete
    :param lmn_coordinates: LMN coordinates at the event under consideration
    :param colors: colors to be used in the plotting
    :return:
    """
    origin = [0, 0, 0]
    X, Y, Z = zip(origin, origin, origin)
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])

    ax = fig_name.add_subplot(221, projection='3d', aspect='equal')
    ax.set_xlabel('$X$', rotation=150)
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$', rotation=60)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    for n in range(len(lmn_coordinates)):
        U, V, W = zip(lmn_coordinates[n])
        ax.quiver(X, Y, Z, U, V, W, color=colors[n])
        ax.quiver(X, Y, -1, np.dot(x, lmn_coordinates[n]), np.dot(y, lmn_coordinates[n]), 0, color='#808080')
        ax.quiver(X, 1, Z, np.dot(x, lmn_coordinates[n]), 0, np.dot(z, lmn_coordinates[n]), color='#808080')
        ax.quiver(-1, Y, Z, 0, np.dot(y, lmn_coordinates[n]), np.dot(z, lmn_coordinates[n]), color='#808080')

    ax = fig_name.add_subplot(222, aspect='equal')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    for n in range(len(lmn_coordinates)):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        x_c, y_c = np.dot(x, lmn_coordinates[n]), np.dot(y, lmn_coordinates[n])
        ax.quiver(X, Y, x_c, y_c, color=colors[n], angles='xy', scale_units='xy', scale=1)

    ax = fig_name.add_subplot(223, aspect='equal')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    for n in range(len(lmn_coordinates)):
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        x_c, z_c = np.dot(x, lmn_coordinates[n]), np.dot(z, lmn_coordinates[n])
        ax.quiver(X, Z, x_c, z_c, color=colors[n], angles='xy', scale_units='xy', scale=1)

    ax = fig_name.add_subplot(224, aspect='equal')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    for n in range(len(lmn_coordinates)):
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        y_c, z_c = np.dot(y, lmn_coordinates[n]), np.dot(z, lmn_coordinates[n])
        ax.quiver(Y, Z, y_c, z_c, color=colors[n], angles='xy', scale_units='xy', scale=1)


def plot_vectors_2d_3d(event: List[np.ndarray], weird: List[np.ndarray]):
    fig1 = plt.figure(1, figsize=(15, 10))
    colors = ['#ff0000', '#000066', '#006600'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_2d_3d(fig1, event, colors)

    fig2 = plt.figure(2, figsize=(15, 10))
    colors = ['#ff6666', '#3366ff', '#00ff00'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_2d_3d(fig2, weird, colors)

    plt.show()


def plot_current_sheet(event: List[np.ndarray], weird: List[np.ndarray], event_date: datetime, weird_date: datetime,
                       probe: int):
    """
    Plots the current sheets and the spacecraft trajectory between them
    :param event: LMN coordinates for event
    :param weird: LMN coordinates for weird event
    :param event_date: event date
    :param weird_date: weird event date
    :param probe: 1 or 2 for Helios 1 or 2
    :return:
    """
    if weird_date < event_date:
        start = weird_date
        first, end = weird, event
        print('EVENT IN MAGENTA, WEIRD IN BLUE')
        future = False
    else:
        start = event_date
        first, end = event, weird
        print('EVENT IN BLUE, WEIRD IN MAGENTA')
        future = True
    try:
        start_date = start - timedelta(hours=1)
        imported_data = HeliosData(start_date=start_date.strftime('%d/%m/%Y'), start_hour=start_date.hour, duration=3,
                                   probe=probe)
    except ValueError:
        imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=3,
                                   probe=probe)

    imported_data.create_processed_column('vp_magnitude')
    t = np.abs((weird_date - event_date).total_seconds())
    v = np.mean(imported_data.data.loc[start: start + timedelta(seconds=t), 'vp_magnitude'])
    distance = t * v
    print('distance', distance)
    print('theoretical distance to X-line: ', distance / (2 * np.tan(0.5)))

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d', aspect='equal')
    plt.title(str(event_date))
    ax.set_xlabel('$X$', rotation=150)
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$', rotation=60)

    normal_1, normal_2 = first[2], end[2]
    d = find_d_from_distance(distance, normal_2)
    xx, yy = np.meshgrid(np.arange(0, 3 * distance, np.int(distance)) - 1.5 * distance,
                         np.arange(0, 3 * distance, np.int(distance)) - 1.5 * distance)
    z1 = (-normal_1[0] * xx - normal_1[1] * yy) * 1. / normal_1[2]
    if future:
        z2 = (-normal_2[0] * xx - normal_2[1] * yy - d) * 1. / normal_2[2]
    else:
        z2 = (-normal_2[0] * xx - normal_2[1] * yy + d) * 1. / normal_2[2]

    ax.plot_surface(xx, yy, z1, alpha=0.2, color='b')
    ax.plot_surface(xx, yy, z2, alpha=0.5, color='m')

    starting_position = [0, 0, 0]
    trajectory = find_spacecraft_trajectory(imported_data, t, start, starting_position, future)
    x = [pos[0] for pos in trajectory]
    y = [pos[1] for pos in trajectory]
    z = [pos[2] for pos in trajectory]
    ax.scatter(x, y, z)
    ax.scatter(x[0], y[0], z[0], color='k')
    ax.scatter(x[9], y[9], z[9], color='k')
    distance_plane_to_point = (normal_2[0] * x[-1] + normal_2[1] * y[-1] + normal_2[2] * z[-1] + d) / np.sqrt(
        normal_2[0] ** 2 + normal_2[1] ** 2 + normal_2[2] ** 2)
    print('distance from 2: ', distance_plane_to_point)

    b_and_v_plotting(ax, imported_data, event_date, weird_date, starting_position, future, event, weird)
    add_m_n_vectors(ax, event, weird, distance, future)

    plt.show()


def add_m_n_vectors(ax, event: list, weird: list, distance: float, future: bool):
    if future:
        X, Y, Z = zip([0, 0, 0], [0, 0, 0], [0, 0, 0])
    else:
        X, Y, Z = zip([-distance, 0, 0], [-distance, 0, 0], [-distance, 0, 0])
    U, V, W = zip(event[1])
    ax.quiver(X, Y, Z, U, V, W, color='r', length=distance, normalize=True)
    U, V, W = zip(event[2])
    ax.quiver(X, Y, Z, U, V, W, color='r', length=distance, normalize=True)
    U, V, W = zip(event[0])
    ax.quiver(X, Y, Z, U, V, W, color='k', length=distance, normalize=True)

    if not future:
        X, Y, Z = zip([0, 0, 0], [0, 0, 0], [0, 0, 0])
    else:
        X, Y, Z = zip([distance, 0, 0], [distance, 0, 0], [distance, 0, 0])
    U, V, W = zip(weird[1])
    ax.quiver(X, Y, Z, U, V, W, color='r', length=distance, normalize=True)


def find_spacecraft_trajectory(imported_data: HeliosData, t: float, start: datetime, starting_position: List[float],
                               future: bool):
    trajectory = [np.array(starting_position)]
    pos_x, pos_y, pos_z = starting_position[0], starting_position[1], starting_position[2]
    default_vx = np.mean(imported_data.data.loc[start: start + timedelta(seconds=t), 'vp_x'])
    default_vy = np.mean(imported_data.data.loc[start: start + timedelta(seconds=t), 'vp_y'])
    default_vz = np.mean(imported_data.data.loc[start: start + timedelta(seconds=t), 'vp_z'])
    time_split = t / 10
    for loop in range(15):
        v_x = -np.mean(imported_data.data.loc[start: start + timedelta(seconds=time_split), 'vp_x'])
        v_y = -np.mean(imported_data.data.loc[start: start + timedelta(seconds=time_split), 'vp_y'])
        v_z = -np.mean(imported_data.data.loc[start: start + timedelta(seconds=time_split), 'vp_z'])
        if np.isnan(v_x) or np.isnan(v_y) or np.isnan(v_z):  # usually if one is nan the others are too
            v_x, v_y, v_z = default_vx, default_vy, default_vz
        if future:
            pos_x -= time_split * v_x
            pos_y -= time_split * v_y
            pos_z -= time_split * v_z
        else:
            pos_x += time_split * v_x
            pos_y += time_split * v_y
            pos_z += time_split * v_z
        trajectory.append(np.array([pos_x, pos_y, pos_z]))
        start = start + timedelta(seconds=time_split)
    return trajectory


def find_d_from_distance(distance, normal2):
    """
    Finds the d component of the ax+by+cz=d of a given plane when the first plane passes though origin
    Spacecraft passes though origin as well and moves nearly only in x direction
    :param distance: distance that the spacecraft travels before encountering plane 2
    :param normal2: normal of plane 2
    :return:
    """
    # point = [0,0,0] belongs to plane 1, so [distance, 0, 0] belong to plane 2, equation ax+by+cz+d==0
    # distance is mostly in x, so we assume y and z will not noticeably change
    d = - normal2[0] * distance
    return d


def b_and_v_plotting(ax, imported_data: HeliosData, event_time: datetime, weird_time: datetime,
                     starting_position: list, future: bool, event: list, weird: list):
    """
    Plots the b and v fields on the current sheet plot
    :param ax: ax of the figure on which the vectors are going to be plotted
    :param imported_data: HeliosDate
    :param event_time: time of event
    :param weird_time: time of other event
    :param starting_position: position where the spacecraft starts
    :param future: false if weird is before event, true otherwise
    :return:
    """
    mag_field, v_field = [], []
    x, y, z = [], [], []
    if weird_time < event_time:
        start_time, end_time = weird_time, event_time
    else:
        start_time, end_time = event_time, weird_time
    t = (end_time - start_time).total_seconds()
    if event_time < weird_time:
        vel = np.array([-np.mean(imported_data.data.loc[start_time - timedelta(seconds=t):start_time, 'vp_x']),
                        -np.mean((imported_data.data.loc[start_time - timedelta(seconds=t):start_time, 'vp_y'])),
                        -np.mean(imported_data.data.loc[start_time - timedelta(seconds=t):start_time, 'vp_z'])])
    else:
        vel = np.array([np.mean(imported_data.data.loc[start_time - timedelta(seconds=t):start_time, 'vp_x']),
                        np.mean((imported_data.data.loc[start_time - timedelta(seconds=t):start_time, 'vp_y'])),
                        np.mean(imported_data.data.loc[start_time - timedelta(seconds=t):start_time, 'vp_z'])])

    # pos_x, pos_y, pos_z = starting_position[0], starting_position[1], starting_position[2]
    pos_x, pos_y, pos_z = starting_position[0] + vel[0] * t, starting_position[1] + vel[1] * t, starting_position[2] + \
                          vel[2] * t
    time_split = t / 10
    start = start_time - timedelta(seconds=t)
    for loop in range(30):
        default_vx = np.mean(imported_data.data.loc[start: start + timedelta(seconds=t), 'vp_x'])
        default_vy = np.mean(imported_data.data.loc[start: start + timedelta(seconds=t), 'vp_y'])
        default_vz = np.mean(imported_data.data.loc[start: start + timedelta(seconds=t), 'vp_z'])

        x.append(pos_x)
        y.append(pos_y)
        z.append(pos_z)
        b = np.array([np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'Bx']),
                      np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'By']),
                      np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'Bz'])])
        _v = np.array([-np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'vp_x']),
                       -np.mean((imported_data.data.loc[start:start + timedelta(seconds=time_split), 'vp_y'])),
                       -np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'vp_z'])])
        vs = np.array([-np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'vp_x']),
                       np.mean((imported_data.data.loc[start:start + timedelta(seconds=time_split), 'vp_y'])),
                       np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'vp_z'])])
        if np.isnan(_v[0]) or np.isnan(_v[1]) or np.isnan(_v[2]):
            _v = np.array([-default_vx, -default_vy, -default_vz])
        t1 = np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'Tp_par'])
        t2 = np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'Tp_perp'])
        n = np.mean(imported_data.data.loc[start:start + timedelta(seconds=time_split), 'n_p'])
        if future:
            pos_x -= time_split * _v[0]
            pos_y -= time_split * _v[1]
            pos_z -= time_split * _v[2]
        else:
            pos_x += time_split * _v[0]
            pos_y += time_split * _v[1]
            pos_z += time_split * _v[2]

        mag_field.append(b)
        v_field.append(_v)
        start += timedelta(seconds=time_split)
        b_lmn = np.array([np.dot(b, event[0]), np.dot(b, event[1]), np.dot(b, event[2])])
        v_lmn = np.array([np.dot(vs, event[0]), np.dot(vs, event[1]), np.dot(vs, event[2])])
        if not np.isnan(b[0]):
            print(start, b_lmn, v_lmn, np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2),
                  np.sqrt(_v[0] ** 2 + _v[1] ** 2 + _v[2] ** 2), t1, t2, n)

    u, v, w = [b[0] for b in mag_field], [b[1] for b in mag_field], [b[2] for b in mag_field]
    a, b, c = [_v[0] for _v in v_field], [_v[1] for _v in v_field], [_v[2] for _v in v_field]

    for n in range(len(u)):
        if (np.isnan(u[n]) or np.isnan(v[n]) or np.isnan(w[n])) and n != 0 and n != len(u) - 1:
            u[n] = np.mean([u[n - 1], u[n + 1]])
            v[n] = np.mean([v[n - 1], v[n + 1]])
            w[n] = np.mean([w[n - 1], w[n + 1]])
    vec_length = 4 * (x[1] - x[0])
    ax.quiver(x, y, z, u, v, w, color='g', length=1.5 * vec_length, normalize=True)
    ax.quiver(x, y, z, a, b, c, color='k', length=0.5 * vec_length, normalize=True)


def plot_possible_folded_sheet(normal: List[np.ndarray], distances: List[float]):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('$X$', rotation=150)
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$', rotation=60)
    xx, yy = np.meshgrid(np.arange(0, 7 * sum(distances), 7 * np.int(sum(distances)) - 1) - 3.5 * sum(distances),
                         np.arange(0, 3 * sum(distances), 3 * np.int(sum(distances)) - 1) - 1.5 * sum(distances))
    z = (-normal[0][0] * xx - normal[0][1] * yy) * 1. / normal[0][2]
    ax.plot_surface(xx, yy, z, alpha=0.4, color='b')
    ax.scatter(0, 0, 0, color='k')
    colors = ['m', 'g', 'r', 'y', 'c', 'k']
    for n in range(len(distances)):
        d = - normal[n + 1][0] * sum(distances[:n + 1])
        z = (-normal[n + 1][0] * xx - normal[n + 1][1] * yy - d) * 1. / normal[n + 1][2]
        ax.plot_surface(xx, yy, z, alpha=0.4, color=colors[n])
        ax.scatter(sum(distances[:n + 1]), 0, 0, color='k')

    ax.plot([-4 * sum(distances), 4 * sum(distances)], [0, 0], [0, 0])
    plt.show()


def compare_magnitude(event_date: datetime, weird_date: datetime, probe: int):
    """
    Plots the b and v fields magnitudes, without the current sheet or spacecraft (quicker than plotting them)
    :param event_date: date of event
    :param weird_date: date of other event
    :param probe: 1 or 2 for Helios 1 or 2
    :return:
    """
    if weird_date < event_date:
        start_time, end_time = weird_date, event_date
    else:
        start_time, end_time = event_date, weird_date
    imported_data = HeliosData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_time.hour, duration=1,
                               probe=probe)

    t = (end_time - start_time).total_seconds()
    mag_field = []
    v_field = []
    for loop in range(10):
        b = np.array([np.mean(imported_data.data.loc[start_time:start_time + timedelta(seconds=t / 5), 'Bx']),
                      np.mean(imported_data.data.loc[start_time:start_time + timedelta(seconds=t / 5), 'By']),
                      np.mean(imported_data.data.loc[start_time:start_time + timedelta(seconds=t / 5), 'Bz'])])
        _v = np.array([np.mean(imported_data.data.loc[start_time:start_time + timedelta(seconds=t / 5), 'vp_x']),
                       np.mean((imported_data.data.loc[start_time:start_time + timedelta(seconds=t / 5), 'vp_y'])),
                       np.mean(imported_data.data.loc[start_time:start_time + timedelta(seconds=t / 5), 'vp_z'])])

        mag_field.append(b)
        v_field.append(_v)
        start_time += timedelta(seconds=t / 5)
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    bx, by, bz = [b[0] for b in mag_field], [b[1] for b in mag_field], [b[2] for b in mag_field]
    vx, vy, vz = [b[0] for b in v_field], [b[1] for b in v_field], [b[2] for b in v_field]
    y, z = np.zeros(len(bx)), np.zeros(len(bx))
    x = np.arange(0, 4 * len(bx), 4)
    ax.quiver(x, y, z, bx, by, bz, length=6, normalize=True, color='m')
    ax.quiver(x, y, z, vx, vy, vz, length=3, normalize=True, color='b')
    ax.scatter([len(bx), -1], [len(bx), -1], [len(bx), -len(bx)], color='k')

    plt.show()


if __name__ == '__main__':
    # crossing = {'ev_date': datetime(1974, 12, 15, 8, 32), 'w_date': datetime(1974, 12, 15, 8, 37), 'probe': 1}
    # crossing = {'ev_date': datetime(1978, 4, 22, 10, 31), 'w_date': datetime(1978, 4, 22, 10, 10), 'probe': 2,
    #             'comment': 'nearly parallel lines'}
    # crossing = {'ev_date': datetime(1977, 12, 4, 7, 13), 'w_date': datetime(1977, 12, 4, 7, 20), 'probe': 2,
    #             'comment': 'normal reconnection, with turbulent bm'}
    #
    crossing = {'ev_date': datetime(1976, 12, 1, 6, 12), 'w_date': datetime(1976, 12, 1, 5, 48), 'probe': 1,
                'comment': 'maybe three times crossing the same sheet? but no reconnection event :('}
    crossing = {'ev_date': datetime(1976, 12, 1, 6, 12), 'w_date': datetime(1976, 12, 1, 6, 23), 'probe': 1,
                'comment': 'maybe three times crossing the same sheet? but no reconnection event :('}
    crossing = {'ev_date': datetime(1976, 12, 1, 6, 23), 'w_date': datetime(1976, 12, 1, 7, 16), 'probe': 1,
                'comment': 'maybe three times crossing the same sheet? but no reconnection event :('}
    crossing = {'ev_date': datetime(1976, 12, 1, 7, 16), 'w_date': datetime(1976, 12, 1, 7, 31), 'probe': 1,
                'comment': 'maybe three times crossing the same sheet? but no reconnection event :('}

    # crossing = {'ev_date': datetime(1976, 12, 6, 6, 3), 'w_date': datetime(1976, 12, 6, 6, 37), 'probe': 2,
    #             'comment': 'normal reconnection with turbulent field? maybe 3 crossings?'}
    # crossing = {'ev_date': datetime(1976, 12, 6, 6, 37), 'w_date': datetime(1976, 12, 6, 6, 57), 'probe': 2,
    #             'comment': 'normal reconnection with turbulent field? maybe three crossings?'}
    # crossing = {'ev_date': datetime(1976, 1, 30, 6, 26), 'w_date': datetime(1976, 1, 30, 6, 40), 'probe': 2,
    #             'comment': 'needs more analysis and very hard to see anything'}
    # crossing = {'ev_date': datetime(1975, 10, 31, 14, 42), 'w_date': datetime(1975, 10, 31, 14, 44), 'probe': 1,
    #             'comment': 'maybe not enough data for this one'}
    # crossing = {'ev_date': datetime(1977, 2, 3, 5, 20), 'w_date': datetime(1977, 2, 3, 4, 27), 'probe': 1,
    #             'comment': 'not totally parallel reconnection'}

    # crossing = {'ev_date': datetime(1975, 3, 19, 0, 44), 'w_date': datetime(1975, 3, 19, 0, 14), 'probe': 1,
    #             'comment': 'close to sun'}

    # crossing = {'ev_date': datetime(1977, 10, 30, 21, 4), 'w_date': datetime(1977, 10, 30, 20, 27), 'probe': 1,
    #             'comment': 'stable reconnection'}
    #
    # crossing = {'ev_date': datetime(1976, 1, 29, 0, 4), 'w_date': datetime(1976, 1, 28, 23, 48), 'probe': 2}

    ev_date, w_date, probe = crossing['ev_date'], crossing['w_date'], crossing['probe']
    a, b = compare_lmn(ev_date, w_date, probe, 5, 1)
    print(a)
    print(b)
    # rotation = get_rotation_matrix(a, b)
    # rotation_matrix_to_euler(rotation)
    # plot_vectors_2d_3d(a, b)

    plot_current_sheet(a, b, ev_date, w_date, probe)

    # 06/12/1976
    # plot_possible_folded_sheet([np.array([0.39828885, 0.26623724, -0.87777202]),
    #                              np.array([-0.47328068, -0.27463772, 0.83700629]),
    #                              np.array([0.46363021, 0.36476725, -0.80746014])], [814118.333738, 481761.213457])

    # 01/12/1976
    # plot_possible_folded_sheet([np.array([ 0.53842085,  0.83226682,  0.13204142]),
    #                              np.array([ 0.82988484, -0.36531417,  0.42170691]),
    #                              np.array([-0.93136905, -0.02975537, -0.36285852])], [452062.781834, 207985.190599])
    # plot_possible_folded_sheet([np.array([0.53842085, 0.83226682, 0.13204142]),
    #                             np.array([0.82988484, -0.36531417, 0.42170691]),
    #                             np.array([-0.93136905, -0.02975537, -0.36285852]),
    #                             np.array([-0.06672256, 0.7672766, -0.63783596]),
    #                             np.array([0.55796194, -0.59920598, 0.57413472])],
    #                            [452062.781834, 207985.190599, 1005439.93777, 287286.345991])
