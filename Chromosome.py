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

    def gaussian_mutation(self, lower_index, upper_index):
        #For each value between the index
        for i in range(lower_index, upper_index):
            random_sample = self._random.normal()  # Adding a value from a Cauchy distribution

            # Clamp the value within 3 standard deviation
            # (99.7% values drawn from a normal distribution are within 3 standard deviation)
            if (random_sample > 3.0):
                random_sample = 3.0
            elif (random_sample < -3.0):
                random_sample = -3.

            #Rescale the sample so that the value correspond to 1% of the total range
            new_value = 0.001 * (((random_sample - (-3.0)) * (self._upper_bound[i] - self._lower_bound[i]))/(3.0 - (-3.0))
                        +self._lower_bound[i])
            new_value = self._array_of_genes[i] + new_value

            #Make sure we don't go out of bound
            if (new_value > self._upper_bound[i]):
                new_value = self._upper_bound[i]
            elif (new_value < self._lower_bound[i]):
                new_value = self._lower_bound[i]

            self._array_of_genes[i] = new_value