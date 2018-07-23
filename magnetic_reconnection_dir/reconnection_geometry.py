import csv
from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u

from data_handler.data_importer.helios_data import HeliosData
from magnetic_reconnection_dir.lmn_coordinates import hybrid_mva


def compare_lmn(event_date: datetime, weird_thing_date: datetime, probe: int):
    L, M, N = hybrid_mva(event_date, probe)
    L_w, M_w, N_w = hybrid_mva(weird_thing_date, probe)

    print('dot products', np.dot(L, M), np.dot(L, N), np.dot(M, N))
    print('dot products', np.dot(L_w, M_w), np.dot(L_w, N_w), np.dot(M_w, N_w))

    return [L, M, N], [L_w, M_w, N_w]


def get_rotation_matrix(event: List[np.ndarray], weird: List[np.ndarray]) -> np.ndarray:
    # rotation * event = weird
    rotation = np.matmul(np.matrix(weird), inv(np.matrix(event)))
    print('det', np.linalg.det(rotation))
    return rotation


def rotation_matrix_to_euler(rotation: np.ndarray) ->np.ndarray:
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

    print(np.degrees(np.array([theta, psi, phi])))
    return np.array([theta, psi, phi])


def plot_vectors(event: List[np.ndarray], weird: List[np.ndarray]):
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    origin = [0, 0, 0]
    X, Y, Z = zip(origin, origin, origin)
    colors = ['#ff0000', '#000066', '#006600'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    for n in range(len(event)):
        U, V, W = zip(event[n])
        ax.quiver(X, Y, Z, U, V, W, color=colors[n])
    colors = ['#ff6666', '#3366ff', '#00ff00'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    for n in range(len(event)):
        U, V, W = zip(weird[n])
        ax.quiver(X, Y, Z, U, V, W, color=colors[n])
    plt.show()


if __name__ == '__main__':
    a, b = compare_lmn(datetime(1978, 4, 22, 10, 31), datetime(1978, 4, 22, 10, 10), 2)
    # a, b = compare_lmn(datetime(1974, 12, 15, 8, 32), datetime(1974, 12, 15, 8, 37), 1)
    print(a)
    print(b)
    print(get_rotation_matrix(a, b))
    rotation = get_rotation_matrix(a, b)
    rotation_matrix_to_euler(rotation)
    plot_vectors(a, b)
