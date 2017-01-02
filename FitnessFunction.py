import math
import numpy as np
import random


def calculate_fitness(fun_fitness, *args):
    return fun_fitness(*args)


# Between -500 and 500
def schwefel_function(array_genes):
    ndim = len(array_genes)
    # We use the Schwefel function slightly modified so it becomes a minimization problem
    value = np.sum(array_genes * np.sin(np.sqrt(np.abs(array_genes))))
    return (418.98288727243374296449474059045314788818359375*float(ndim)) - value


# Between -5 and 10 for x
# Between 0 and 15 for y
def branin(x):
    x_0 = x[0]
    x_1 = x[1]
    value = np.square(x_1 - (5.1/(4.0*np.square(math.pi)))*np.square(x_0) + (5.0/math.pi)*x_0 - 6.0) + \
        10.0*(1-(1./(8*math.pi)))*np.cos(x_0) + 10.0
    return value


# Between -100 and 100
def schaffer_function(array_genes):
    array = np.square(array_genes[:-1]) + np.square(array_genes[1:])
    value = np.dot(np.power(array, 0.25), np.square(np.sin(50 * np.power(array, 0.1))) + 1)
    return value


# Between -5.12 and 5.12
def rastrigin(array_genes):
    ndim = len(array_genes)
    value = np.dot(array_genes, array_genes) - 10 * np.sum(np.cos(2 * math.pi * array_genes))
    return value + 10*ndim


# Between -1.28 and 1.28
def noise_function(array_genes):
    value = np.dot(np.arange(1, array_genes+1), np.power(array_genes, 4))
    return value + random.random()


# Between -10 and 10
def schwefel_func_p1_dot_2_unimodal(array_genes):
    cumsum_array = np.cumsum(array_genes)
    value = np.dot(cumsum_array, cumsum_array)
    return value


# Between -10 and 10
def rosenbrock(array_genes):
    array_one = np.square(array_genes[:-1]) - array_genes[1:]
    array_two = array_genes[:-1] - 1
    value = 100 * np.dot(array_one, array_one) + np.dot(array_two, array_two)
    return value


# Between -100 and 100
def elliptic_function(array_genes):
    ndim = len(array_genes)
    value = np.dot(np.power(1000000, np.arange(ndim) / (ndim - 1)), np.square(array_genes))
    return value


def sphere_function(array_genes):
    value = np.dot(array_genes, array_genes)
    return value


# Between -32 and 32
def ackley_function(array_genes):
    ndim = len(array_genes)
    value = 20 - 20 * math.exp(-0.2 * math.sqrt(1.0/ndim * np.dot(array_genes, array_genes)))
    value += math.e - math.exp(1.0/ndim * np.sum(np.cos(2 * math.pi * array_genes)))
    return value


# Between -500 and 500
def weierstrass_function(array_genes):
    ndim = len(array_genes)
    a_powers_array = np.power(0.5, np.arange(21))
    b_powers_array = np.power(3.0, np.arange(21)).reshape(21, 1)
    value = np.sum(a_powers_array * np.cos(2 * math.pi * np.dot(b_powers_array, array_genes.reshape(1, ndim) + 0.5).T))
    value -= ndim * np.sum(a_powers_array * np.cos(math.pi * b_powers_array))
    return value


# Between -600 and 600
def griewank_function(array_genes):
    ndim = len(array_genes)
    value = 1 + 1.0 / 4000 * np.dot(array_genes, array_genes) - np.prod(np.cos(array_genes / np.sqrt(1+np.arange(ndim))))
    return value