'''
Handle the Chromosome, that is the representation of the input to the fitness function and the
mutation operator for the GA.
'''

class Chromosome(object):
    def __init__(self, number_of_genes, lower_bound, upper_bound, random):
        self._number_of_genes = number_of_genes
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._random = random
        self._array_of_genes = []

        #Randomly initialize the Chromosomes using an uniform distribution between the lower and upper bound
        for i in range(self._number_of_genes):
            random_genes = self._random.random_sample() * (self._upper_bound[i] - self._lower_bound[i]) + \
                           self._lower_bound[i]
            self._array_of_genes.append(random_genes)

    def get_array_genes(self):
        return self._array_of_genes

    def get_gene(self, index):
        return self._array_of_genes[index]