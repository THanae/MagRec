import heliopy.data.spice as spice_data
import heliopy.spice as spice
from datetime import datetime, timedelta
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support


# load the kernel
def kernel_loader(spacecraft=2):
    orbiter_kernel = spice_data.get_kernel('helios' + str(spacecraft))
    spice.furnish(orbiter_kernel)
    orbiter = spice.Trajectory('Helios ' + str(spacecraft))
    return orbiter


def orbit_times_generator(start_date='20/01/1976', end_date='01/10/1979', interval=1):
    start_time = datetime.strptime(start_date, '%d/%m/%Y')
    end_time = datetime.strptime(end_date, '%d/%m/%Y')
    times = []
    while start_time < end_time:
        times.append(start_time)
        start_time = start_time + timedelta(days=interval)
    return times


def orbit_generator(orbiter, times, observing_body='Sun', frame='ECLIPJ2000'):
    orbiter.generate_positions(times, observing_body, frame)
    orbiter.change_units(u.au)


def plot_orbit(orbiter, spacecraft=2):
    quantity_support()
    # to have nice different colors depending on time
    times_float = [(t - orbiter.times[0]).total_seconds() for t in orbiter.times]
    fig = plt.figure()
    circle = plt.Circle((0, 0), 0.004, color='r')
    ax = fig.add_subplot(111)
    # could add option to have uniform c
    ax.scatter(orbiter.x, orbiter.y, s=3, c=times_float)
    ax.scatter(orbiter.x[0], orbiter.y[0], s=5, c='b')
    ax.scatter(orbiter.x[10], orbiter.y[10], s=5, c='r')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title(
        'Orbit of Helios ' + str(spacecraft) + '  between ' + str(orbiter.times[0]) + ' and ' + str(orbiter.times[-1]))
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(circle)
    # for 3d:
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(orbiter.x, orbiter.y, orbiter.z, **kwargs)
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)

    plt.show()


def plot_period(orbiter, spacecraft=2):
    quantity_support()
    # to have nice different colors depending on time
    times_float = [(t - orbiter.times[0]).total_seconds() for t in orbiter.times]
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    # could add option to have uniform c
    sun_distance = np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2)
    plt.plot(orbiter.times, sun_distance)
    plt.title(
        'Orbit of Helios ' + str(spacecraft) + '  between ' + str(orbiter.times[0]) + ' and ' + str(orbiter.times[-1]))
    plt.show()


if __name__ == '__main__':
    orbiter = kernel_loader(2)
    times = orbit_times_generator()
    orbit_generator(orbiter, times)
    radius = np.sqrt(orbiter.x ** 2 + orbiter.y ** 2 + orbiter.z ** 2)
    # print(orbiter.times)
    print('the aphelion is ', np.min(radius), ' at ', orbiter.times[np.argmin(radius)])
    print('the perihelion is ', np.max(radius), ' at ', orbiter.times[np.argmax(radius)])
    # plot_orbit(orbiter)
    plot_period(orbiter)
