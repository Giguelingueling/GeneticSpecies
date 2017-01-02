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

        self._fitness_before_real_evalutation = []

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

        self._bound_distance = self.create_bound_distance()

    def create_bound_distance(self):
        bound_distance = []
        for i in range(len(self._lower_bound)):
            bound_distance.append(self._upper_bound[i] - self._lower_bound[i])
            bound_distance = np.array(bound_distance)
        return bound_distance

    def run_swarm_process(self):
        # First we find the combination of creature that cover the space the more thoroughly.
        # To achieve that, we use KMEANS with k=2 on the list of creature position.
        kmeans = KMeans(n_clusters=2)

        swarm_positions = self._swarm.get_list_position()  # Get the list of point in the space for KMeans
        # Normalize the dimension of each point so the point chosen is irrelevant of the possible different unities
        # of each dimensions

        normalized_swarm_position = np.array(swarm_positions) / np.array(self._bound_distance)
        kmeans.fit(normalized_swarm_position)  # Train KMeans
        centers = kmeans.cluster_centers_  # Get the centers
        centers *= self._bound_distance  # Go back to the original dimension
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

        self._swarm.add_creature_to_swarm(position=creature0_position)
        self._swarm.add_creature_to_swarm(position=creature1_position)

        # Add the creatures position and fitness to the list of position and fitness evaluated
        self._list_real_evaluation_position.append(creature0_position)
        self._list_real_evaluation_fitness.append(creature0_fitness)
        self._list_real_evaluation_position.append(creature1_position)
        self._list_real_evaluation_fitness.append(creature1_fitness)

        # Train the regressor
        self._regressor.train_regressor(self._list_real_evaluation_position, self._list_real_evaluation_fitness)

        # From here, we alternate between exploration and exploitation randomly based on an heuristic except for the
        # Very first pass where we for the algorithm to be in exploration mode for one more evaluation
        # (3 evaluations total)
        self.exploration()
        self._number_of_real_evaluation -= 1  # Did a real evaluation

        # Now that we have three points evaluated, we are ready to start the algorithm for the requested amount of real
        # evaluations. Or until the user stop the program
        for generation in range(self._number_of_real_evaluation):
            print self._list_real_evaluation_position
            print self._list_real_evaluation_fitness
            print self._fitness_before_real_evalutation
            # Decide if we explore or exploite.
            exploitation_threshold = max(0.2, 1/math.sqrt((generation+2)/2))
            exploitation_threshold = 0.0
            if self._random.rand() < exploitation_threshold:
                best_creature_ever = self.exploration()

                # TODO once the exploitation algorithm is finish. Remove the if/else and move this block after
                # We are almost done with this generation, get the real value of the point of interest found
                new_point_to_add_fitness = FitnessFunction.calculate_fitness(self._fitness_function,
                                                                             best_creature_ever.get_position(),
                                                                             self._number_of_dimensions)
                # Finish the generation by adding the new creature to the list and updating the regressor
                self._list_real_evaluation_position.append(best_creature_ever.get_position())
                self._list_real_evaluation_fitness.append(new_point_to_add_fitness)

                self._fitness_before_real_evalutation.append(self._regressor.get_estimation_fitness_value(best_creature_ever.get_position())[0])
                choose_kernel = False
                if len(self._list_real_evaluation_fitness) % 10 == 0:
                    choose_kernel = True
                self._regressor.train_regressor(self._list_real_evaluation_position,
                                                self._list_real_evaluation_fitness, choose_kernel=choose_kernel)
                print "Smallest point found: ", new_point_to_add_fitness, "Fitness found by the PSO:", \
                    best_creature_ever.get_fitness(), " At position: ", best_creature_ever.get_position()
                # Reset swarm fitness
                self._swarm.reset_swarm()
            else:
                best_creature_ever = self.exploitation()


        index = self._list_real_evaluation_fitness.index(min(self._list_real_evaluation_fitness))
        print "Smallest point found: ", self._list_real_evaluation_fitness[index], " At position: ", \
            self._list_real_evaluation_position[index]

    def exploration(self):
        print "EXPLORATION"
        # We want to get EI
        self._regressor.set_EI_bool(True)
        # We want to get the curiosity
        self._swarm.set_curiosity(True)
        # Make sure that every creature has been evaluated
        best_fitness = min(self._list_real_evaluation_fitness)
        print "BEST CURRENT FITNESS", best_fitness
        self._swarm.evaluate_fitness_swarm(fitness_function=self._regressor, best_real_function_value=best_fitness)

        # run swarm optimization with number of iterations.
        best_creature_ever = self._swarm.run_swarm_optimization(max_iterations=self._number_of_generation_swarm,
                                                                function_to_optimize=self._regressor,
                                                                inertia_factor=self._inertia_factor,
                                                                self_confidence=self._self_confidence,
                                                                swarm_confidence=self._swarm_confidence,
                                                                sense_of_adventure=self._sense_of_adventure,
                                                                best_real_function_value=best_fitness,
                                                                list_position_with_real_fitness=
                                                                self._list_real_evaluation_position)
        return best_creature_ever

    def exploitation(self):
        print "EXPLOITATION"
        # Finish exploration by updating the regressor
        # We don't want to get EI
        self._regressor.set_EI_bool(False)

        seed_position = []
        fitness_seed_position = []
        # Determine the number of parallel optimization to perform on already evaluated position
        if len(self._list_real_evaluation_position) <= 10:
            # Number of parallel optimization = number of real evaluation.
            seed_position.extend(self._list_real_evaluation_position)
            fitness_seed_position.extend(self._list_real_evaluation_fitness)
        else:
            # Create 10 parallel optimization with seed on the 10 best points found.
            # Get the 3 best positions and chose randomly 7 other points
            # Start by partially sorting the array to get the best 3 points
            index_partially_sorted = np.argpartition(a=self._list_real_evaluation_fitness, kth=3)[0:3]
            seed_position = self._list_real_evaluation_position[index_partially_sorted]
            fitness_seed_position = self._list_real_evaluation_fitness[index_partially_sorted]

            # Add randomly 7 positions different from the 3 positions already selected
            index_range = range(len(self._list_real_evaluation_fitness))
            # remove the already selected values
            for i in index_partially_sorted:
                index_range.remove(i)
            choices = np.random.choice(index_range, 7, replace=False)
            seed_position.extend(self._list_real_evaluation_position[choices])
            fitness_seed_position.extend(self._list_real_evaluation_position[choices])

        # Reduce the number of creatures per swarms so that the sum of all the creatures in all of the swarms
        # Are equivalent to when we are in exploration mode.
        size_of_swarm = int(self._swarm_size / len(seed_position))
        swarms = []
        limit = math.pi/10.0
        normalized_list_real_evaluation = np.array(self._list_real_evaluation_position)/np.array(self._bound_distance)
        fitness_list = np.array(self._list_real_evaluation_fitness)
        # Calculate the std of the fitness
        fitness_std = np.std(fitness_list)
        # If the std is very small (close to machine error), then the fitness information is basically uniform.
        # This mean it's useless to try to exploite this area. Launch and exploration phase instead.
        if fitness_std < np.finfo(float).eps*10:
            return self.exploration()


        # Get the sorted list from biggest value to smallest of the position fitness.
        sorted_fitness_list = np.flipud(np.sort(fitness_list))
        for position, fitness_position in zip(seed_position, fitness_seed_position):
            # We need to set the upper and lower bound in function of the seed point for this swarm and the variance
            # of the gaussian process.
            normalized_position = np.array(np.array(position)/self._bound_distance)
            '''
            # old way
            difference_between_position_and_points_to_avoid = normalized_list_real_evaluation - normalized_position
            distance_normalized_position_other_points = np.apply_along_axis(
                np.linalg.norm, axis=1, arr=difference_between_position_and_points_to_avoid)

            closest_point = np.argmin(distance_normalized_position_other_points)
            closesness_factor = 0.5

            difference_vector = closest_point-position

            if fitness_position == sorted_fitness_list[0]:
                mean = abs(sorted_fitness_list[0]-sorted_fitness_list[1])
            else:
                mean = 1.0
            '''

            # Bounding box is defined as when the std of the gaussian process is equal or smaller to pi/10.0*fitness_std
            # around the current position considered. However, since we cannot know analytically the std of the GP,
            # we have to use another way to actually define the measurement of the bounding box.
            # To do so, we check the closest known position with a real fitness to the creature we're working on right
            # now (i.e. identified by the name position). We then trace a segment between those two points and check
            # in the middle of this segment if the std of the GP is above or below our threshold.
            # If it's below: take half the original segment and call it L. We add L to the current segment and verify
            # the GP std. We keep increasing the segment by L until the GP std become bigger than our threshold. This is
            # our new boundary
            # If it's above: We divide by sqrt(2) the length of the segment and we check again. We keep repeating that
            # process until we get a value that is below the researched threshold. This is our new bounding box.

            # First, find the closest point by calculating the distance between the current position and all the others
            # with known fitness and get the min
            index = 0
            min_distance = float('Inf')
            min_distance_index = 0
            print normalized_list_real_evaluation
            print normalized_position
            for point in normalized_list_real_evaluation:
                if np.array_equal(normalized_position, point) == False:
                    distance = np.linalg.norm(normalized_position-point)
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_index = index
                index += 1

            closest_point = normalized_list_real_evaluation[min_distance_index]
            print normalized_list_real_evaluation
            # Now that we have the closest point, calculate the middle point between the closest point and the
            # considered position
            middle_point = (normalized_position+closest_point)/2.
            # Get the variance in the GP of this point
            prediction, gp_std = self._regressor.get_estimation_fitness_value(middle_point)
            max_std = (2.0*math.pi)/10.
            if gp_std > max_std:
                # get a smaller and smaller segment until we find a point that is within our boundary
                new_segment = np.array(middle_point-normalized_position)
                shrinking_factor = 1.0
                while gp_std > max_std:
                    point_to_evaluate = np.array(normalized_position) + (shrinking_factor*new_segment)
                    prediction, gp_std = self._regressor.get_estimation_fitness_value(point_to_evaluate *
                                                                                      np.array(self._bound_distance))
                    shrinking_factor /= math.sqrt(2)
            else:
                # get a bigger and bigger segment that grow with the length of the middle point each time
                # calculate the segment to add each time
                segment_to_add = (np.array(middle_point)-np.array(normalized_position))/2.0
                new_segment = np.array(middle_point)
                while gp_std < max_std:
                    new_segment += np.array(segment_to_add)
                    prediction, gp_std = self._regressor.get_estimation_fitness_value(new_segment *
                                                                                      np.array(self._bound_distance))

            # Now that the new segment is calculated. We simply get the distance between the position we're interested
            # in and this point we just found (new_segment)
            distance_bound = np.linalg.norm(new_segment-normalized_position)

            # Take care of the curse of dimensionality
            distance_bound /= math.pow(math.sqrt(2), self._number_of_dimensions-1)

            # Unormalize
            distance_bound *= self._bound_distance

            # We create an array with the same dimension as the function dimension. We have to provide an upper bound
            # and a lower bound which will both have a distance of the norm we just calculated (distance_bound)
            upper_bound_exploitation = position + (distance_bound * np.ones(self._number_of_dimensions))
            lower_bound_exploitation = position - (distance_bound * np.ones(self._number_of_dimensions))

            swarm_to_add = Swarm(swarm_size=size_of_swarm, number_of_dimensions=self._number_of_dimensions,
                                 lower_bound=lower_bound_exploitation, upper_bound=upper_bound_exploitation,
                                 random=self._random)
            swarms.append(swarm_to_add)


        self._regressor.train_regressor(self._list_real_evaluation_position, self._list_real_evaluation_fitness)
        return 0.0
# Schwefel
lower_bound = np.array([-500.0])
upper_bound = np.array([500.0])
# Rastrigin
# lower_bound = np.array([-5.12, -5.12])
# upper_bound = np.array([5.12, 5.12])
# Branin
# lower_bound = np.array([-5.0, 0.0])
# upper_bound = np.array([10.0, 15.0])
number_of_dimensions = 1
number_of_real_evaluation = 100
swarm_size = 100
number_of_generation_swarm = 100
swarmProcess = SwarmProcess(lower_bound=lower_bound, upper_bound=upper_bound, number_of_dimensions=number_of_dimensions,
                            number_of_real_evaluation=number_of_real_evaluation, swarm_size=swarm_size,
                            number_of_generation_swarm=number_of_generation_swarm,
                            fitness_function=FitnessFunction.schwefel_function, inertia_factor=0.5, self_confidence=1.5,
                            swarm_confidence=1.5, sense_of_adventure=1.5)
swarmProcess.run_swarm_process()