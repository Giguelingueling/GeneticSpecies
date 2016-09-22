

class Creature(object):
    def __init__(self, ID):
        self._ID = ID
        self._age = 0
        #For the fitness, lower is better.
        self.fitness = float('Inf')#Since we minimize function, we initialize the fitness with the highest possible value