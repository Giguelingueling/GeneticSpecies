import math
import numpy as np
import random


def calculate_fitness(fun_fitness, *args):
    return fun_fitness(*args)


# Between -500 and 500
def schwefel_function(array_genes):
    # We use the Schwefel function slightly modified so it becomes a minimization problem
    summation = np.sum(array_genes * np.sin(np.sqrt(np.abs(array_genes))))
    return (418.98288727243374296449474059045314788818359375*float(len(array_genes))) - summation


# Between -5 and 10 for x
# Between 0 and 15 for y
def branin(x):
    x_0 = x[0]
    x_1 = x[1]
    y = np.square(x_1 - (5.1/(4.0*np.square(math.pi)))*np.square(x_0) + (5.0/math.pi)*x_0 - 6.0) + \
        10.0*(1-(1./(8*math.pi)))*np.cos(x_0) + 10.0

    result = y

    return result


# Between -100 and 100
def schaffer_function(array_genes):
    summation = 0.0
    for i in range(len(array_genes)-1):
        summation += math.pow(math.pow(array_genes[i], 2) + math.pow(array_genes[i + 1], 2), 0.25) * \
                     (math.pow(math.sin(50 * math.pow(math.pow(array_genes[i], 2) +
                                                      math.pow(array_genes[i + 1], 2), 0.10)), 2) + 1.0)
    return summation


# Between -5.12 and 5.12
def rastrigin(array_genes):
    summation = np.dot(array_genes, array_genes) - 10 * np.sum(np.cos(2 * math.pi * array_genes))
    return summation + 10*len(array_genes)


# Between -1.28 and 1.28
def noise_function_(array_genes):
    value = np.dot(np.arange(1, array_genes+1), np.power(array_genes, 4))
    return value + random.random()


# Between -10 and 10
def schwefel_func_p1_dot_2_unimodal(array_genes):
    value = 0.0
    for i in range(len(array_genes)):
        value_to_be_squared = 0.0
        for j in range(i):
            value_to_be_squared += array_genes[j]
        value += np.square(value_to_be_squared)
    return value


# Between -10 and 10
def rosenbrock(array_genes):
    value = 0.0
    for i in range(len(array_genes)-1):
        value += 100.0*np.square((array_genes[i+1] - np.square(array_genes[i]))) + np.square(array_genes[i] - 1)
    return value
