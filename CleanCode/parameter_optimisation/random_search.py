from typing import List, Union
import numpy as np

from CleanCode.parameter_optimisation.mcc_finding import mcc_from_parameters


def fitness_function(parameters: dict) -> List[Union[float, dict]]:
    """
    Calculates the fitness function for a given set of parameters
    :param parameters: parameters to be tested
    :return: mcc associated with parameters
    """
    mcc = mcc_from_parameters(parameters)
    if np.isnan(mcc[0]):
        mcc[0] = 0
    return mcc


def random_algorithm(maximum_iterations: int = 100) -> List[Union[float, dict]]:
    """
    Finds the best possible parameters by randomly generating parameters
    :param maximum_iterations: maximum number of iterations before the code stops running
    :return: best mcc obtained, with associated parameters
    """
    initial_parameters = {'xyz': {'sigma_sum': np.random.uniform(1.9, 3.1), 'sigma_diff': np.random.uniform(1.9, 3.1),
                                  'minutes_b': np.random.uniform(1, 10), 'minutes': np.random.uniform(1, 10)},
                          'lmn': {'minimum_walen': np.random.uniform(0.6, 1),
                                  'maximum_walen': np.random.uniform(1, 1.4)}}
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
    radius = max_radius + 1
    new_parameters = {}
    while radius > max_radius:
        new_parameters = {'xyz': {'sigma_sum': np.random.uniform(1.9, 3.1), 'sigma_diff': np.random.uniform(1.9, 3.1),
                                  'minutes_b': np.random.uniform(1, 10), 'minutes': np.random.uniform(1, 10)},
                          'lmn': {'minimum_walen': np.random.uniform(0.6, 1),
                                  'maximum_walen': np.random.uniform(1, 1.4)}}
        radius = radius_of_parameters(parameters, new_parameters)
    return new_parameters


def radius_of_parameters(parameters1: dict, parameters2: dict) -> float:
    """
    Finds the distance between two sets of parameters in the parameter hyperspace
    Moves through the parameter hyperspace by finding the best MCC possible
    :param parameters1: center set of parameters
    :param parameters2: outer set of parameters
    :return: distance between the two sets of parameters
    """
    radius = 0
    params_1 = {**parameters1['xyz'], **parameters1['lmn']}
    params_2 = {**parameters2['xyz'], **parameters2['lmn']}
    print(params_2)
    for key in params_1.keys():
        radius += (params_1[key] - params_2[key]) ** 2
    radius = np.sqrt(radius)
    return radius


if __name__ == '__main__':
    mcc1 = random_algorithm()
