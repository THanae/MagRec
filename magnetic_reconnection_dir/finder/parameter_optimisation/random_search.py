from typing import List, Union
import numpy as np

from magnetic_reconnection_dir.finder.correlation_finder import CorrelationFinder
from magnetic_reconnection_dir.finder.parameter_optimisation.mcc_calculations import mcc_from_parameters


def fitness_function(parameters: dict) -> List[Union[float, dict]]:
    """
    Calculates the fitness function for a given set of parameters
    :param parameters: parameters to be tested
    :return: mcc associated with parameters
    """
    mcc = mcc_from_parameters(parameters, CorrelationFinder())
    if np.isnan(mcc[0]):
        mcc[0] = 0
    return mcc


def random_algorithm(maximum_iterations: int = 100) -> List[Union[float, dict]]:
    """
    Finds the best possible parameters by randomly generating parameters
    :param maximum_iterations: maximum number of iterations before the code stops running
    :return: best mcc obtained, with associated parameters
    """
    initial_parameters = {'sigma_sum': np.random.uniform(1.9, 3.1), 'sigma_diff': np.random.uniform(1.9, 3.1),
                          'minutes_b': np.random.uniform(1, 10), 'minutes': np.random.uniform(1, 10),
                          'minimum walen': np.random.uniform(0.6, 1), 'maximum walen': np.random.uniform(1, 1.4)}
    current_parameters = initial_parameters.copy()
    current_iteration = 0
    mcc, max_mcc = [0, {}], [0, {}]
    mcc_calculated = False
    while current_iteration < maximum_iterations and mcc[0] < 0.9:
        if mcc_calculated:
            mcc = max_mcc
        else:
            mcc = fitness_function(current_parameters)
            max_mcc = mcc.copy()
        mcc_calculated = False
        previous_parameters = current_parameters.copy()
        current_parameters = parameters_from_hypersphere(current_parameters)
        challenging_mcc = fitness_function(current_parameters)
        if challenging_mcc[0] > max_mcc[0]:
            max_mcc = challenging_mcc
            current_iteration += 1
        else:
            current_parameters = previous_parameters.copy()
            mcc_calculated = True
            current_iteration += 0.25
        print(f'Iteration {current_iteration}: MCC {max_mcc[0]} with parameters {current_parameters}')
    return max_mcc


def parameters_from_hypersphere(parameters: dict, max_radius: float = 2) -> dict:
    """
    Generates set of parameters within the sphere of radius max_radius, centered at parameters
    :param parameters: parameters at the center of the sphere
    :param max_radius: radius of the sphere in the parameters hyperspace
    :return: new set of parameters within the sphere
    """
    radius = 100
    new_parameters = {}
    while radius > max_radius:
        new_parameters = {'sigma_sum': np.random.uniform(1.9, 3.1), 'sigma_diff': np.random.uniform(1.9, 3.1),
                          'minutes_b': np.random.uniform(1, 10), 'minutes': np.random.uniform(1, 10),
                          'minimum walen': np.random.uniform(0.6, 1), 'maximum walen': np.random.uniform(1, 1.4)}
        radius = radius_of_parameters(parameters, new_parameters)
    return new_parameters


def radius_of_parameters(parameters1: dict, parameters2: dict) -> float:
    """
    Finds the distance between two sets of parameters in the parameter hyperspace
    :param parameters1: center set of parameters
    :param parameters2: outer set of parameters
    :return: distance between the two sets of parameters
    """
    radius = 0
    for key in parameters1.keys():
        radius += (parameters1[key] - parameters2[key]) ** 2
    radius = np.sqrt(radius)
    return radius


if __name__ == '__main__':
    mcc1 = random_algorithm()
    # mcc2 = random_algorithm()
    # mcc3 = random_algorithm()

    # best value so far
    # MCC 0.7377111135633175
    # {'sigma_sum': 2.3855388099995887, 'sigma_diff': 2.9083100667153365, 'minutes_b': 4.916354076654231,
    # 'minutes': 6.746805442993062, 'minimum walen': 0.9875196009186451, 'maximum walen': 1.1388006476154153}
