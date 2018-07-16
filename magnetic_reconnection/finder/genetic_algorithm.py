from datetime import timedelta, datetime

from data_handler.imported_data import ImportedData
from magnetic_reconnection.finder.correlation_finder import CorrelationFinder

import numpy as np
import matplotlib.pyplot as plt

# lists [event, probe, number of reconnections]
event_list = [[datetime(1974, 12, 15, 14, 0, 0), 1, 1], [datetime(1974, 12, 15, 20, 0, 0), 1, 1],
              [datetime(1975, 1, 18, 13, 0, 0), 1, 1], [datetime(1975, 2, 7, 1, 0, 0), 1, 1],
              [datetime(1975, 9, 22, 3, 30, 0), 1, 1], [datetime(1975, 12, 19, 21, 0, 0), 1, 1],
              [datetime(1976, 1, 19, 6, 0, 0), 2, 1], [datetime(1976, 1, 27, 7, 0, 0), 2, 1],
              [datetime(1976, 1, 30, 2, 0, 0), 2, 2], [datetime(1976, 3, 4, 9, 0, 0), 2, 1],
              [datetime(1976, 12, 15, 1, 0, 0), 2, 1], [datetime(1977, 4, 5, 22, 0, 0), 2, 1],
              [datetime(1978, 1, 25, 7, 0, 0), 2, 1], [datetime(1978, 2, 26, 4, 0, 0), 2, 1],
              [datetime(1977, 4, 23, 3, 0, 0), 2, 1], [datetime(1977, 12, 17, 1, 0, 0), 1, 1],
              [datetime(1978, 3, 17, 16, 0, 0), 1, 1], [datetime(1979, 6, 21, 2, 0, 0), 1, 1],
              [datetime(1980, 1, 3, 20, 0, 0), 1, 1], [datetime(1980, 1, 16, 14, 0, 0), 1, 1],

              [datetime(1976, 1, 18, 6, 0, 0), 2, 0], [datetime(1976, 2, 2, 7, 0, 0), 2, 0],
              [datetime(1977, 4, 22, 3, 0, 0), 2, 0], [datetime(1976, 2, 4, 7, 0, 0), 2, 0],
              [datetime(1976, 3, 5, 9, 0, 0), 2, 0], [datetime(1976, 12, 16, 1, 0, 0), 2, 0],
              [datetime(1977, 4, 6, 22, 0, 0), 2, 0], [datetime(1977, 12, 19, 1, 0, 0), 2, 0],
              [datetime(1978, 1, 5, 10, 0, 0), 2, 0], [datetime(1974, 12, 17, 14, 0, 0), 1, 0],
              [datetime(1974, 12, 17, 20, 0, 0), 1, 0], [datetime(1975, 1, 19, 13, 0, 0), 1, 0],
              [datetime(1975, 2, 8, 1, 0, 0), 1, 0], [datetime(1975, 9, 24, 3, 30, 0), 1, 0],
              [datetime(1975, 12, 20, 21, 0, 0), 1, 0], [datetime(1977, 12, 18, 1, 0, 0), 1, 0],
              [datetime(1978, 3, 22, 16, 0, 0), 1, 0], [datetime(1976, 12, 1, 2, 0, 0), 1, 0],
              [datetime(1980, 1, 4, 20, 0, 0), 1, 0], [datetime(1980, 1, 18, 14, 0, 0), 1, 0]
              ]


def genetic_algorithm(genes, first_population_size=10, best_samples_size=3, randomly_chosen_sample_size=2,
                      number_of_descendants=3, mutation_probability=0.1, event_list_split=20, iterations=20):
    # not a genetic algorithm yet, still in implementation phase
    # fitness score will be mcc

    population = generate_first_population(first_population_size, genes)
    performances = []
    for loop in range(iterations):
        print('GENERATION', loop)
        np.random.shuffle(event_list)
        sorted_by_performance = performance_per_gene(population, event_list_split)
        performances.append(sorted_by_performance[0])
        print('performance', sorted_by_performance)
        next_generation = selection(sorted_by_performance, best_samples=best_samples_size,
                                    randomly_chosen_samples=randomly_chosen_sample_size)
        descendants = crossover(next_generation, descendants_number=number_of_descendants, best_genes=best_samples_size)
        population = mutation(descendants, mutation_probability)

    def get_key(item):
        return item[0]

    best_mcc = [performance[0] for performance in performances]
    plt.plot(best_mcc)
    plt.show()

    # puts nans at the end of the performance list
    performances_no_nans = [perf for perf in performances if not np.isnan(perf[0])]
    performances_nans = [perf for perf in performances if np.isnan(perf[0])]
    return sorted(performances_no_nans, key=get_key, reverse=True) + performances_nans


def fitness(gene, event_list_split):
    """
    Calculates the mcc for a given gene
    :param gene: gene of a given population
    :param event_list_split: number of events that are considered in the mcc calculation
    :return: mcc
    """
    # test on random part of the data
    # this could avoid over-fitting and allow more iterations of the algorithm in less time
    events = event_list[:event_list_split]
    f_n, t_n, t_p, f_p = 0, 0, 0, 0
    sigma_s, sigma_d, mins_b = gene[0], gene[1], gene[2]
    for event, probe, reconnection_number in events:
        interval = 3
        start_time = event - timedelta(hours=interval / 2)
        start_hour = event.hour
        data = ImportedData(start_date=start_time.strftime('%d/%m/%Y'), start_hour=start_hour,
                            duration=interval, probe=probe)
        reconnection = CorrelationFinder.find_magnetic_reconnections(self=CorrelationFinder(),
                                                                     imported_data=data,
                                                                     sigma_sum=sigma_s,
                                                                     sigma_diff=sigma_d,
                                                                     minutes_b=mins_b)

        if reconnection_number == 0:
            if len(reconnection) == 0:
                t_n += 1
            else:
                f_p += len(reconnection)
        else:
            if len(reconnection) < reconnection_number:
                f_n += reconnection_number - len(reconnection)
                t_p += len(reconnection)
            elif len(reconnection) == reconnection_number:
                t_p += len(reconnection)
            else:  # more detected than real
                f_p += len(reconnection) - reconnection_number
                t_p += reconnection_number
    mcc = (t_p * t_n + f_n * f_p) / np.sqrt((t_p + f_p) * (t_p + f_n) * (t_n + f_p) * (t_n + f_n))
    return mcc


def pair_generation(genes):
    sigma_sum, sigma_diff, minutes_b = genes[0], genes[1], genes[2]
    sigma_s = (np.max(sigma_sum) - np.min(sigma_sum)) * np.random.random_sample() + np.min(sigma_sum)
    sigma_d = (np.max(sigma_diff) - np.min(sigma_diff)) * np.random.random_sample() + np.min(sigma_diff)
    mins_b = (np.max(minutes_b) - np.min(minutes_b)) * np.random.random_sample() + np.min(minutes_b)
    return [sigma_s, sigma_d, mins_b]


# for now, the following code follows the reasoning of the genetic algorithm tutorial available at
# https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
# but adapted for this particular use
def generate_first_population(population_size, genes):
    population = []
    for n in range(population_size):
        population.append(pair_generation(genes))
    return population


def performance_per_gene(population, event_list_split):
    """
    Finds the mcc performance of the given gens
    :param population: all the genes that are being considered
    :return: list of list of mcc and genes
    """
    performance = []
    for gene in population:
        performance.append([fitness(gene, event_list_split), gene])

    def get_key(item):
        return item[0]

    # puts nans at the end of the performance list
    performance_no_nans = [perf for perf in performance if not np.isnan(perf[0])]
    performance_nans = [perf for perf in performance if np.isnan(perf[0])]
    return sorted(performance_no_nans, key=get_key, reverse=True) + performance_nans


def selection(sorted_population, best_samples, randomly_chosen_samples):
    """
    Takes the best performing individuals as well as some randomly chosen individuals
    :param sorted_population: list of mcc performance and genes, sorted by best performance
    :param best_samples: number of best performing samples we want to take
    :param randomly_chosen_samples: number of random samples we want to choose
    :return: list of parents of genes that are going to be tested in the next generation
    """
    next_generation = []
    for n in range(best_samples):
        next_generation.append(sorted_population[n][1])
    for n in range(randomly_chosen_samples):
        next_generation.append(sorted_population[np.random.randint(0, len(sorted_population) - 1)][1])

    return next_generation


def create_descendant(gene1, gene2):
    """
    Creates a child from two genes
    :param gene1: parent gene
    :param gene2: parent gene
    :return: child gene
    """
    descendant = []
    for n in range(len(gene1)):
        if np.random.rand() < 0.5:
            descendant.append(gene1[n])
        else:
            descendant.append(gene2[n])
    return descendant


def crossover(genes, descendants_number, best_genes):
    """
    Create children from parent genes
    :param genes: parent genes (chosen from the previous population)
    :param descendants_number: number of children we want to create for parent genes
    :return:
    """
    next_population = []
    for n in range(best_genes):
        next_population.append(genes[n])
    np.random.shuffle(genes)
    # can do in len(genes) and have random parents but might take longer
    for n in range(len(genes)):
        # for n in range(int(len(genes) / 2)):
        for m in range(descendants_number):
            next_population.append(
                create_descendant(genes[n], genes[len(genes) - 1 - np.random.randint(0, len(genes))]))
            # create_descendant(genes[n], genes[len(genes) - 1 - n]))
    return next_population


def mutate_gene(gene):
    """
    Slightly changes the gene, hoping to get better results
    :param gene: gene that we want to change
    :return: mutated gene
    """
    modification = np.int(np.random.rand() * len(gene))
    if modification == 0:
        gene = gene
    else:
        if np.random.rand() < 0.5:
            gene[modification] = gene[modification] + np.random.rand()
        else:
            gene[modification] = gene[modification] - np.random.rand()
    return gene


def mutation(genes, probability):
    """
    Mutates the given genes with a given probability
    :param genes: this generation's genes, that might be mutated
    :param probability: probability that the genes will be mutated
    :return: mutated population
    """
    for n in range(len(genes)):
        if np.random.rand() < probability:
            genes[n] = mutate_gene(genes[n])
    return genes


if __name__ == '__main__':
    sigma_sum = np.arange(1, 4, 0.5)
    sigma_diff = np.arange(1, 4, 0.5)
    minutes_b = [3, 4, 5, 6, 7, 8]
    genetic_algorithm([sigma_sum, sigma_diff, minutes_b])
