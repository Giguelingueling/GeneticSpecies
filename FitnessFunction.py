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


# Between -50 and 50
def generalized_penalized_function(array_genes):
    ndim = len(array_genes)
    array = 1 + 1/4 * (array_genes + 1)
    value = np.dot(np.square(array[:-1]-1), 1 + 10 * np.square(np.sin(math.pi * array[1:])))
    value += np.square(np.sin(math.pi * array[0])) + np.square(array[-1]-1)
    value *= math.pi / ndim

    array = np.zeros(ndim)
    array[array_genes > 10] = 100 * np.power(array[array > 10] - 10, 4)
    array[array_genes < -10] = 100 * np.power(-array[array < -10] - 10, 4)
    value += np.sum(array)
    return value


# Between -500 and 500. Using CLPSO's version of this function.
def schwefel_function_rotated(array_genes, rotation_matrix):
    ndim = len(array_genes)
    array_genes_rotated = np.dot(rotation_matrix, array_genes - 420.96) + 420.96
    array = np.zeros(ndim)
    less_indices = np.abs(array_genes_rotated) <= 500
    more_indices = np.abs(array_genes_rotated) > 500
    array[less_indices] = array_genes_rotated[less_indices] * np.sin(np.sqrt(np.abs(array_genes_rotated[less_indices])))
    array[more_indices] = 0.001 * np.square(np.abs(array_genes_rotated[more_indices]) - 500)
    value = np.sum(array)
    return (418.98288727243374296449474059045314788818359375*float(ndim)) - value


# Returns the rotation matrix of dimension ndim x ndim. Somewhat heavy computationally.
def get_rotation_matrix(ndim):
    matrix = np.eye(ndim)
    matrix[:, 0] = 1
    Q, _ = np.linalg.qr(matrix)
    return Q


# Example of how to use the relevant shifted/rotated functions
def example(ndim):
    # Generate rotation matrix once and for all.
    rotation_matrix = get_rotation_matrix(ndim)

    # Generate shift vector once and for all. Random according to boundaries of
    # solution space, or fixed.
    shift = np.random.uniform(low=0, high=1, size=ndim)

    # Let's pretend the following loop is a full PSO run.
    for i in range(1000):
        # Just a point to evaluate.
        x = np.random.uniform(low=0, high=1, size=ndim)

        # Rotated Schwefel. This one is coded explicitely because it's a special snowflake.
        value = schwefel_function_rotated(x, rotation_matrix)

        # Rotated Rosenbrock.
        value += rosenbrock(np.dot(rotation_matrix, x))

        # Rotated Rastrigin.
        value += rastrigin(np.dot(rotation_matrix, x))

        # Shifted Rosenbrock.
        value += rosenbrock(x - shift)

        # Shifted Rastrigin.
        value += rastrigin(x - shift)

        # Shifted and rotated Rastrigin.
        value += rastrigin(np.dot(rotation_matrix, x - shift))






