import numpy as np
from Swarm import Swarm
import FitnessFunction
from sklearn.cluster import KMeans
import math
from FunctionEstimator import FunctionEstimator

class SwarmProcess(object):
    '''
    lower_bound and upper_bound refer to the min and max of the function to optimize respectively.
    number_of_genes correspond to the number of dimensions of the function to optimize.
    population_size Number of creature in a population
    number_of_generation correspond to the number of the main loop the algorithm will do
    '''
    def __init__(self, lower_bound, upper_bound, number_of_dimensions, number_of_real_evaluation, swarm_size,
                 number_of_generation_swarm, fitness_function, inertia_factor=0.5, self_confidence=1.5,
                 swarm_confidence=1.5, sense_of_adventure=1.5):
        self._number_of_real_evaluation = number_of_real_evaluation
        self._random = np.random.RandomState()
        self._number_of_dimensions = number_of_dimensions
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._number_of_generation_swarm = number_of_generation_swarm

        #Swarm hyper-parameters
        self._inertia_factor = inertia_factor
        self._self_confidence = self_confidence
        self._swarm_confidence = swarm_confidence
        self._sense_of_adventure = sense_of_adventure

        # We remove two for the swarm size.
        # Because the first step of the algorithm is to add two creatures which cover the space
        # in the most efficient manner.
        self._swarm_size = swarm_size-2

        self._fitness_function = fitness_function

        self._regressor = FunctionEstimator(get_EI=True)

        # Create the main swarm responsible to explore the function
        self._swarm = Swarm(swarm_size=self._swarm_size, number_of_dimensions=self._number_of_dimensions,
                            lower_bound=self._lower_bound, upper_bound=self._upper_bound, random=self._random)

        self._list_real_evaluation_position = []
        self._list_real_evaluation_fitness = []


    def run_swarm_process(self):
        # First we find the combination of creature that cover the space the more thoroughly.
        # To achieve that, we use KMEANS with k=2 on the list of creature position.
        kmeans = KMeans(n_clusters=100)

        swarm_positions = self._swarm.get_list_position()  # Get the list of point in the space for KMeans
        kmeans.fit(swarm_positions)  # Train KMeans
        centers = kmeans.cluster_centers_  # Get the centers
        print "Centers: ", centers

        # Add two new creatures with their position corresponding to the centers of kmeans.
        creature0_position = centers[0]
        creature0_fitness = FitnessFunction.calculate_fitness(self._fitness_function, creature0_position,
                                                              self._number_of_dimensions)
        self._number_of_real_evaluation -= 1  # Did a real evaluation

        creature1_position = centers[1]
        creature1_fitness = FitnessFunction.calculate_fitness(self._fitness_function, creature1_position,
                                                              self._number_of_dimensions)
        self._number_of_real_evaluation -= 1  # Did a real evaluation

        self._swarm.add_creature_to_swarm(creature0_position, creature0_fitness)
        self._swarm.add_creature_to_swarm(creature1_position, creature1_fitness)

        # Add the creatures position and fitness to the list of position and fitness evaluated
        self._list_real_evaluation_position.append(creature0_position)
        self._list_real_evaluation_fitness.append(creature0_fitness)
        self._list_real_evaluation_position.append(creature1_position)
        self._list_real_evaluation_fitness.append(creature1_fitness)

        # Train the regressor
        self._regressor.train(self._list_real_evaluation_position, self._list_real_evaluation_fitness)

        # From here, we alternate between exploration and exploitation randomly based on an heuristic except for the
        # Very first pass where we for the algorithm to be in exploration mode for one more evaluation
        # (3 evaluations total)
        self.exploration()
        self._number_of_real_evaluation -= 1  # Did a real evaluation

        # Now that we have three points evaluated, we are ready to start the algorithm for the requested amount of real
        # evaluations. Or until the user stop the program
        for generation in range(self._number_of_real_evaluation):
            # Decide if we explore or exploite.
            exploitation_threshold = max(0.2, 1/math.sqrt((generation+2)/2))
            if self._random.rand() < exploitation_threshold:
                self.exploration()
            else:
                self.exploitation()

    def exploration(self):
        print "EXPLORATION"
        # We want to get EI
        self._regressor.set_EI_bool(True)
        # We want to get the curiosity
        self._swarm.set_curiosity(True)
        # run swarm optimization with number of iterations.
        self._swarm.run_swarm_optimization(self._number_of_generation_swarm, self._regressor, self._inertia_factor,
                                           self._self_confidence, self._swarm_confidence, self._sense_of_adventure)
        # Finish exploration by updating the regressor
        self._regressor.update_regressor(self._list_real_evaluation_position, self._list_real_evaluation_fitness)

        #Reset swarm fitness
        self._swarm.reset_swarm()

    def exploitation(self):
        print "EXPLOITATION"
        # Finish exploration by updating the regressor
        # We don't want to get EI
        self._regressor.set_EI_bool(False)
        #We don't want to allow curiosity
        self._swarm.set_curiosity(False)

        self._regressor.update_regressor(self._list_real_evaluation_position, self._list_real_evaluation_fitness)

lower_bound = np.array([-500.0, -500.0])
upper_bound = np.array([500.0, 500.0])
number_of_dimensions = 2
number_of_real_evaluation = 50
swarm_size = 1000
number_of_generation_swarm = 100
swarmProcess = SwarmProcess(lower_bound=lower_bound, upper_bound=upper_bound, number_of_dimensions=number_of_dimensions,
                            number_of_real_evaluation=number_of_real_evaluation, swarm_size=swarm_size,
                            number_of_generation_swarm=number_of_generation_swarm,
                            fitness_function=FitnessFunction.schwefel_function)
swarmProcess.run_swarm_process()