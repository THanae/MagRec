import csv
from datetime import datetime, timedelta
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u
from tqdm import tqdm

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
    # rotation * event = weird
    rotation = np.matmul(np.matrix(weird), inv(np.matrix(event)))
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


def plot_2d_3d(fig_name: matplotlib.figure.Figure, lmn_coordinates: List[np.ndarray], colors: list):
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

    ax = fig_name.add_subplot(221, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    for n in range(len(lmn_coordinates)):
        U, V, W = zip(lmn_coordinates[n])
        ax.quiver(X, Y, Z, U, V, W, color=colors[n])
        ax.quiver(X, Y, -1, np.dot(x, lmn_coordinates[n]), np.dot(y, lmn_coordinates[n]), 0, color='#808080')
        ax.quiver(X, 1, Z, np.dot(x, lmn_coordinates[n]), 0, np.dot(z, lmn_coordinates[n]), color='#808080')
        ax.quiver(-1, Y, Z, 0, np.dot(y, lmn_coordinates[n]), np.dot(z, lmn_coordinates[n]), color='#808080')

    ax = fig_name.add_subplot(222)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    for n in range(len(lmn_coordinates)):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        x_c = np.dot(x, lmn_coordinates[n])
        y_c = np.dot(y, lmn_coordinates[n])
        ax.quiver(X, Y, x_c, y_c, color=colors[n], angles='xy', scale_units='xy', scale=1)

    ax = fig_name.add_subplot(223)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    for n in range(len(lmn_coordinates)):
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        x_c = np.dot(x, lmn_coordinates[n])
        z_c = np.dot(z, lmn_coordinates[n])
        ax.quiver(X, Z, x_c, z_c, color=colors[n], angles='xy', scale_units='xy', scale=1)

    ax = fig_name.add_subplot(224)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    for n in range(len(lmn_coordinates)):
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        y_c = np.dot(y, lmn_coordinates[n])
        z_c = np.dot(z, lmn_coordinates[n])
        ax.quiver(Y, Z, y_c, z_c, color=colors[n], angles='xy', scale_units='xy', scale=1)


def plot_vectors_2d_3d(event: List[np.ndarray], weird: List[np.ndarray]):
    fig1 = plt.figure(1)
    colors = ['#ff0000', '#000066', '#006600'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_2d_3d(fig1, event, colors)

    fig2 = plt.figure(2)
    colors = ['#ff6666', '#3366ff', '#00ff00'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_2d_3d(fig2, weird, colors)

    plt.show()


def get_starting_position(vec1: List[Tuple], vec2: List[Tuple], d: float):
    """
    Finds the best starting position for the spacecraft (returns the origin if none is found)
    :param vec1: all points of plane 1
    :param vec2: all points of plane 2
    :param d: distance between the planes
    :return:
    """
    plane1_pos = []
    for m in tqdm(range(len(vec1))):
        x = vec1[m][0]
        y = vec1[m][1]
        z = vec1[m][2]
        wanted_x = x + d
        for n in range(len(vec2)):
            if 0.9 * np.abs(vec2[n][0]) < np.abs(wanted_x) < 1.1 * np.abs(vec2[n][0]) and np.sign(wanted_x) == np.sign(vec2[n][0]):
                if 0.8 * np.abs(vec2[n][1]) < y < 1.2 * np.abs(vec2[n][1]) and np.sign(y) == np.sign(vec2[n][1]):
                    if 0.5* np.abs(vec2[n][2]) < z < 1.5 * np.abs(vec2[n][2]) and np.sign(z)== np.sign(vec2[n][2]):
                        print('hurray', wanted_x)
                        plane1_pos.append([x, y, z, vec2[n][2]])
    if plane1_pos:
        z_pos = [np.abs(pos[2] - pos[3]) for pos in plane1_pos]
        arg_pos = np.argmin(z_pos)
        return [plane1_pos[arg_pos][0], plane1_pos[arg_pos][1], plane1_pos[arg_pos][2]]
    else:
        return [0, 0, 0]


def plot_current_sheet(event: List[np.ndarray], weird: List[np.ndarray], event_date: datetime, weird_date: datetime, probe: int):
    """
    Plots the current sheets and the spacecraft trajectory between them
    :param event: LMN coordinates for event
    :param weird: LMN coordinates for weird event
    :param event_date: event date
    :param weird_date: weird event date
    :param probe: 1 or 2 for Helios 1 or 2
    :return:
    """
    scale = 1000  # scaling factor
    t = np.abs((weird_date - event_date).total_seconds())
    if weird_date < event_date:
        start = weird_date
    else:
        start = event_date

    imported_data = HeliosData(start_date=start.strftime('%d/%m/%Y'), start_hour=start.hour, duration=1, probe=probe)
    imported_data.create_processed_column('vp_magnitude')
    v = np.mean(imported_data.data['vp_magnitude'])
    d = t * v / scale

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    normal_event = event[2]
    normal_weird = weird[2]
    xx, yy = np.meshgrid(np.arange(0, 3*d, 50) - d / 2, np.arange(0, 3*d, 50) - d / 2)
    z1 = (-normal_event[0] * xx - normal_event[1] * yy) * 1. / normal_event[2]
    if weird_date < event_date:
        z2 = (-normal_weird[0] * xx - normal_weird[1] * yy +d) * 1. / normal_weird[2]
    else:
        z2 = (-normal_weird[0] * xx - normal_weird[1] * yy - d) * 1. / normal_weird[2]
    ax.plot_surface(xx, yy, z1, alpha=0.2, color='b')
    ax.plot_surface(xx, yy, z2, alpha=0.2, color='m')

    vec1, vec2 = list(zip(*(xx.flat, yy.flat, z1.flat))), list(zip(*(xx.flat, yy.flat, z2.flat)))
    starting_position = get_starting_position(vec1, vec2, d)
    trajectory = [np.array(starting_position)]
    pos_x, pos_y, pos_z = starting_position[0], starting_position[1], starting_position[2]
    v_x = np.mean(imported_data.data['vp_x'])
    v_y = np.mean(imported_data.data['vp_y'])
    v_z = np.mean(imported_data.data['vp_z'])
    for loop in range(80):
        if weird_date < event_date:
            pos_x += (t / 50) * v_x
            pos_y += (t / 50) * v_y
            pos_z += (t / 50) * v_z
        else:
            pos_x -= (t / 50) * v_x
            pos_y -= (t / 50) * v_y
            pos_z -= (t / 50) * v_z
        trajectory.append(np.array([pos_x, pos_y, pos_z]))
    x = [pos[0] / scale for pos in trajectory]
    y = [pos[1] / scale for pos in trajectory]
    z = [pos[2] / scale for pos in trajectory]
    ax.scatter(x, y, z)
    plt.show()


if __name__ == '__main__':
    # ev_date = datetime(1978, 4, 22, 10, 31)
    ev_date = datetime(1974, 12, 15, 8, 32)
    # w_date = datetime(1978, 4, 22, 10, 10)
    w_date = datetime(1974, 12, 15, 8, 37)
    # probe = 2
    probe = 1
    a, b = compare_lmn(ev_date, w_date, probe, 5, 1)
    print(a)
    print(b)
    # rotation = get_rotation_matrix(a, b)
    # print(rotation)
    # rotation_matrix_to_euler(rotation)
    plot_vectors_2d_3d(a, b)

    # plot_current_sheet(a, b, ev_date, w_date, probe)
