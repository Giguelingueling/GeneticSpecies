from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern  # As suggested by Larochelle et al.
from scipy.stats import norm
import numpy as np
import math

class FunctionEstimator(object):

    def __init__(self, get_EI=False):
        self._regressor = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=4)
        self._get_EI = get_EI

    def train(self, X, y):
        no_error = False
        number_of_try = 0
        while no_error == False:
            no_error = True
            try:
                if number_of_try == 0:
                    number_of_try += 1
                    self._regressor.fit(np.array(X), np.array(y))
                else:
                    self._regressor = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                                               alpha=(math.pow(10, number_of_try/4.0)),
                                                               n_restarts_optimizer=4)

                    self._regressor.fit(np.array(X), np.array(y))
            except Exception:
                print Exception
                no_error = False

    def set_EI_bool(self,EI_bool):
        self._get_EI = EI_bool

    def get_fitness(self, position, best_fitness=0.0):
        if self._get_EI == False:
            return self.get_estimation_fitness_value(position)
        else:
            return self.get_expected_improvement(position, best_fitness)

    def get_estimation_fitness_value(self, position):
        prediction, std = self._regressor.predict(np.atleast_2d(position), return_std=True)
        return prediction[0], std[0]

    def get_expected_improvement(self, position, best_fitness=0.0):
        #TODO write this function to get the EI
        prediction, std = self.get_estimation_fitness_value(position)
        #expected_improvement = norm.cdf((best_fitness-prediction)/std)
        if best_fitness-prediction < 0:
            return (best_fitness - prediction) / std
        else:
            return (best_fitness - prediction) * std

    def update_regressor(self, X, y):
        print "UPDATING REGRESSOR"
        self._regressor.theta0 = self._regressor.kernel_.theta  # Given correlation parameter = MLE

        print len(y)
        self.train(X, y)
