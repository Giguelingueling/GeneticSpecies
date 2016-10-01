from Creature import Creature
import numpy as np


class Swarm(object):

    def __init__(self, swarm_size, number_of_dimensions, lower_bound, upper_bound, random, allow_curiosity=True):
        self._id_creature = 0
        self._random = random
        self._number_of_dimensions = number_of_dimensions
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._swarm_size = swarm_size
        self._swarm_of_creatures = self.create_creatures(swarm_size)
        self._allow_curiosity = allow_curiosity

    def set_curiosity(self, allow_curiosity):
        self._allow_curiosity = allow_curiosity

    def create_creatures(self, swarm_size):
        list_creatures = []
        for i in range(swarm_size):
            list_creatures.append(Creature(id_creature=self._id_creature, number_of_dimensions=self._number_of_dimensions,
                                           lower_bound=self._lower_bound, upper_bound=self._upper_bound,
                                           random=self._random))
            self._id_creature += 1

        return list_creatures

    def add_creature_to_swarm(self, position, fitness):
        position = np.array(position)
        self._swarm_of_creatures.append(Creature(self._id_creature, self._number_of_dimensions, self._lower_bound,
                                        self._upper_bound, self._random, fitness=fitness, position=position))
        self._id_creature += 1
        self._swarm_size += 1

    def reset_swarm(self):
        for creature in self._swarm_of_creatures:
            creature.reset_fitness()
            creature.reset_memory()

    def get_list_position(self):
        list_position = []
        for creature in self._swarm_of_creatures:
            list_position.append(creature.get_position())
        return list_position

    # Get the creature with the lowest fitness in the swarm.
    def get_best_creature(self):
        best_fitness = float('Inf')
        best_creature = self._swarm_of_creatures[0]
        for creature in self._swarm_of_creatures:
            creature_fitness = creature.get_fitness()
            if creature_fitness < best_fitness:
                best_fitness = creature_fitness
                best_creature = creature
        return best_creature

    # Get the lowest fitness ever found so far
    def get_best_ever_fitness(self):
        best_ever_fitness = float('Inf')
        for creature in self._swarm_of_creatures:
            best_creature_fitness = creature.get_best_memory_fitness()
            if best_ever_fitness > best_creature_fitness:
                best_ever_fitness = best_creature_fitness
        return best_ever_fitness

    def run_swarm_optimization(self, number_of_max_iterations, function_to_optimize, inertia_factor, self_confidence,
                               swarm_confidence, sense_of_adventure):
        for i in range(number_of_max_iterations):
            self.update_swarm(fitness_function=function_to_optimize, inertia_factor=inertia_factor,
                              self_confidence=self_confidence, swarm_confidence=swarm_confidence,
                              sense_of_adventure=sense_of_adventure)

    def update_swarm(self, fitness_function, inertia_factor, self_confidence, swarm_confidence,
                     sense_of_adventure):
        # Before updating, we have to find the best creature of the current swarm iteration.
        current_best_creature = self.get_best_creature()
        print current_best_creature.get_fitness(), "   position:  ", current_best_creature.get_position()

        best_ever_fitness = self.get_best_ever_fitness()
        # Now that we have the best creature of the current generation, we're ready to call update on all the creatures
        for creature in self._swarm_of_creatures:
            creature.update_creature(fitness_function=fitness_function, inertia_factor=inertia_factor,
                                     self_confidence=self_confidence, swarm_confidence=swarm_confidence,
                                     creature_adventure_sense=sense_of_adventure,
                                     current_best_creature_position=current_best_creature.get_position(),
                                     allow_curiosity=self._allow_curiosity,
                                     best_ever_creature_fitness=best_ever_fitness)
