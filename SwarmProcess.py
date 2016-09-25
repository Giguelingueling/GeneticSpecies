import numpy as np
from Swarm import Swarm
class SwarmProcess(object):
    '''
    lower_bound and upper_bound refer to the min and max of the function to optimize respectively.
    number_of_genes correspond to the number of dimensions of the function to optimize.
    population_size Number of creature in a population
    number_of_generation correspond to the number of the main loop the algorithm will do
    '''
    def __init__(self, lower_bound, upper_bound, number_of_dimensions, swarm_size, number_of_generation):
        self._total_number_of_generation = number_of_generation
        self._random = np.random.RandomState()
        self._number_of_dimensions = number_of_dimensions
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._swarm_size = swarm_size

        #Create the main swarm responsible to explore the function
        self._swarm = Swarm(swarm_size=self._swarm_size, number_of_dimensions=self._number_of_dimensions,
                            lower_bound=self._lower_bound, upper_bound=self._upper_bound, random=self._random)


    def run_swarm_process(self):
        a=1