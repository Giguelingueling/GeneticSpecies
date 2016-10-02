import math
import numpy as np


def calculate_fitness(fun_fitness, *args):
    return fun_fitness(*args)


# Between -500 and 500
def schwefel_function(array_genes, number_of_genes):
    # We use the Schwefel function slightly modified so it become a minimization problem
    summation = 0.0
    for i in range(number_of_genes):
        summation += array_genes[i] * math.sin(math.sqrt(abs(array_genes[i])))

    return (418.9829*number_of_genes) - summation


# Between -5 and 10 for x
# Between 0 and 15 for y
def branin(x, number_of_genes):
    x_0 = x[0]
    x_1 = x[1]
    y = np.square(x_1 - (5.1/(4.0*np.square(math.pi)))*np.square(x_0) + (5.0/math.pi)*x_0 - 6.0) + \
        10.0*(1-(1./(8*math.pi)))*np.cos(x_0) + 10.0

    result = y

    return result


# Between -100 and 100
def schaffer_function(array_genes, number_of_genes):
    # We use the Schaffer function slightly modified so it become a maximization problem
    # The Maximal value of this function is 0
    summation = 0.0
    for i in range(number_of_genes-1):
        summation += math.pow(math.pow(array_genes[i], 2) + math.pow(array_genes[i + 1], 2), 0.25) * \
                     (math.pow(math.sin(50 * math.pow(math.pow(array_genes[i], 2) +
                                                      math.pow(array_genes[i + 1], 2), 0.10)), 2) + 1.0)
    return summation


# Between -5.12 and 5.12
def rastrigin(array_genes, number_of_genes):
    summation = 0.0
    for i in range(number_of_genes):
        summation += math.pow(array_genes[i], 2) - (10 * math.cos(2 * math.pi * array_genes[i]))
    return 10*number_of_genes+summation
