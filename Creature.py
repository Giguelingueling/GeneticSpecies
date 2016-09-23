
'''
Represent a creature. That is a Chromosome (input of the function) and a fitness (output of the function)
'''
class Creature(object):
    def __init__(self, ID, number_of_dimensions, lower_bound, upper_bound, random):
        self._random = random

        self._ID = ID
        self._age = 0

        #For the fitness, lower is better.
        self.fitness = float('Inf')#Since we minimize function, we initialize the fitness with the highest possible value

        #Array containing the min and max possible for each position
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        self._number_dimensions = number_of_dimensions
        self._position = self.generate_position_random()
        self._memory_best_position = self._position

    #Generate the position of the creature randomly
    def generate_position_random(self):
        position = []
        #Randomly initialize the position using an uniform distribution between the lower and upper bound
        for i in range(self._number_dimensions):
            random_genes = self._random.random_sample() * (self._upper_bound[i] - self._lower_bound[i]) + \
                           self._lower_bound[i]
            position.append(random_genes)
        return position

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