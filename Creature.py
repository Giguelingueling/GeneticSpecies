import Chromosome

'''
Represent a creature. That is a Chromosome (input of the function) and a fitness (output of the function)
'''
class Creature(object):
    def __init__(self, ID):
        self._ID = ID
        self._age = 0
        #For the fitness, lower is better.
        self.fitness = float('Inf')#Since we minimize function, we initialize the fitness with the highest possible value

        self._chromosome = Chromosome

    def get_ID(self):
        return self._ID

    def reset_fitness(self):
        self._fitness = float('Inf')

    def get_fitness(self):
        return self._fitness