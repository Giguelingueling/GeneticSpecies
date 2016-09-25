import numpy as np

'''
Represent a creature. That is a Chromosome (input of the function) and a fitness (output of the function)
'''
class Creature(object):
    def __init__(self, ID, number_of_dimensions, lower_bound, upper_bound, random, fitness_function):
        self._random = random

        self._ID = ID
        self._age = 0

        #Array containing the min and max possible for each position
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        self._number_dimensions = number_of_dimensions
        self._position = self.generate_position_random()
        self._velocity = self.generate_velocity_random()

        # For the fitness, lower is better.
        self.fitness = fitness_function(
            self._position)  # Since we minimize function, we initialize the fitness with the highest possible value

        #Add the variable useful for the creature memory
        self._memory_best_position = self._position
        self._memory_best_fitness = self.fitness

    #Generate the position of the creature randomly
    def generate_position_random(self):
        return self._random.uniform(size=self._number_dimensions) * (self._upper_bound - self._lower_bound) + \
               self._lower_bound

    def generate_velocity_random(self):
        velocity = []
        #Randomly initialize the position using an uniform distribution between the lower and upper bound
        for i in range(self._number_dimensions):
            random_dimension_value = self._random.random_sample() * (self._upper_bound[i] - self._lower_bound[i]) + \
                           self._lower_bound[i]
            velocity.append(random_dimension_value)
        return np.array(velocity)

    #BE CAREFUL WITH BEST CREATURE TO SEND A HARD COPY SO THAT IF IT UPDATE THE POSITION IT DOESN'T CHANGE THE BEST CREATURE POSITION
    def update_velocity(self, inertia_factor, self_confidence, swarm_confidence, creature_adventure_sense,
                        best_creature_position):
        current_motion = inertia_factor*self._velocity
        creature_memory = self_confidence*self._random.rand(self._number_dimensions)*\
                          (self._memory_best_position-self._position)
        swarm_influence = swarm_confidence*self._random.rand(self._number_dimensions)*\
                          (best_creature_position-self._position)

        #TODO find the formula to express the creature curiosity
        #creature_curiosity = creature_adventure_sense*self._random.rand(self._number_dimensions)*

        self._velocity = current_motion+creature_memory+swarm_influence

    def update_position(self):
        self._position = self._position+self._velocity

    def get_ID(self):
        return self._ID

    def reset_fitness(self):
        self._fitness = float('Inf')

    def get_fitness(self):
        return self._fitness

    def get_position(self):
        return self._position

    def reset_memory(self):
        self._memory_best_position = self._position

    def update_creature(self, fitness_function, inertia_factor, self_confidence, swarm_confidence,
                        creature_adventure_sense, best_creature_position):
        #Update velocity and position
        self.update_velocity(inertia_factor, self_confidence, swarm_confidence, creature_adventure_sense,
                             best_creature_position)
        self.update_position()

        #Calculate the fitness
        self._fitness = fitness_function(self._position)

        #If the new fitness is better than its best fitness within memory, update the creature memory
        if(self._fitness < self._memory_best_fitness):
            self._memory_best_fitness = self._fitness
            self._memory_best_position = self._position

        #The creature is now older
        self._age += 1

    def gaussian_mutation(self):
        #For each value between the index
        for i in range(self._number_dimensions):
            random_sample = self._random.normal()  # Adding a value from a Cauchy distribution
            #Rescale the sample so that the value correspond to 1% of the total range and add it to the gene array.
            new_value = self._position[i] + 0.01 * random_sample * (self._upper_bound[i] - self._lower_bound[i])

            #Make sure we don't go out of bound
            if new_value > self._upper_bound[i]:
                new_value = self._upper_bound[i]
            elif new_value < self._lower_bound[i]:
                new_value = self._lower_bound[i]

            self._position[i] = new_value