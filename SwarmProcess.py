import numpy as np
from Swarm import Swarm
import FitnessFunction
from sklearn.cluster import KMeans

class SwarmProcess(object):
    '''
    lower_bound and upper_bound refer to the min and max of the function to optimize respectively.
    number_of_genes correspond to the number of dimensions of the function to optimize.
    population_size Number of creature in a population
    number_of_generation correspond to the number of the main loop the algorithm will do
    '''
    def __init__(self, lower_bound, upper_bound, number_of_dimensions, swarm_size, number_of_generation, fitness_function):
        self._total_number_of_generation = number_of_generation
        self._random = np.random.RandomState()
        self._number_of_dimensions = number_of_dimensions
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # We remove two for the swarm size.
        # Because the first step of the algorithm is to add two creatures which cover the space in the most efficient manner.
        self._swarm_size = swarm_size-2

        self._fitness_function = fitness_function

        #Create the main swarm responsible to explore the function
        self._swarm = Swarm(swarm_size=self._swarm_size, number_of_dimensions=self._number_of_dimensions,
                            lower_bound=self._lower_bound, upper_bound=self._upper_bound, random=self._random,
                            fitness_function=self._fitness_function)


    def run_swarm_process(self):
        list_real_evaluation_position = []
        list_real_evaluation_fitness = []

        #First we find the combination of creature that cover the space the more thoroughly.
        #To achieve that, we use KMEANS with k=2 on the list of creature position.
        kmeans = KMeans(n_clusters=2)

        swarm_positions = self._swarm.get_list_position()#Get the list of point in the space for KMeans
        kmeans.fit(swarm_positions)#Train KMeans
        centers = kmeans.cluster_centers_#Get the centers
        print centers
        print self._swarm.get_list_position()

        #Add two new creatures with their position corresponding to the centers of kmeans.
        creature0_position = centers[0]
        creature0_fitness = FitnessFunction.calculate_fitness(self._fitness_function, creature0_position,
                                                              self._number_of_dimensions)

        creature1_position = centers[1]
        creature1_fitness = FitnessFunction.calculate_fitness(self._fitness_function, creature1_position,
                                                              self._number_of_dimensions)


        self._swarm.add_creature_to_swarm(creature0_position, creature0_fitness)
        self._swarm.add_creature_to_swarm(creature1_position, creature1_fitness)

        print self._swarm.get_list_position()#Get the list of point in the space for KMeans

        #From here, we alternate between exploration and exploitation randomly based on an heuristic except for the
        #Very first pass where we for the algorithm to be in exploration mode for one more evaluation (3 evaluations total)

lower_bound = np.array([-500.0, -500.0])
upper_bound = np.array([500.0, 500.0])
number_of_dimensions = 2
swarm_size = 10
number_of_generation = 100
swarmProcess = SwarmProcess(lower_bound, upper_bound, number_of_dimensions, swarm_size, number_of_generation,
                            FitnessFunction.schwefel_function)
swarmProcess.run_swarm_process()