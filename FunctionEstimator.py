from sklearn.gaussian_process import GaussianProcess
import numpy as np
import math

class FunctionEstimator(object):

    def __init__(self, get_EI=False):
        self._regressor = GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
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
                    self._regressor = GaussianProcess(theta0=math.pow(10,number_of_try)*1e-2, thetaL=1e-4, thetaU=1e-1,
                                                      random_start=100)

                    self._regressor.fit(np.array(X), np.array(y))
            except Exception:
                print Exception
                no_error = False

    def set_EI_bool(self,EI_bool):
        self._get_EI= EI_bool

    def get_fitness(self, position):
        if self._get_EI == False:
            self.get_estimation_fitness_value(position)
        else:
            self.get_expected_improvement(position)

    def get_estimation_fitness_value(self, position):
        prediction, variance = self._regressor.predict(np.atleast_2d(position), eval_MSE=True)
        return prediction[0], variance[0]

    def get_expected_improvement(self, position):
        #TODO write this function to get the EI
        self.get_estimation_fitness_value(position)

    def update_regressor(self, all_creature_data):
        print "UPDATING REGRESSOR"
        if(self._type_of_regressor == 'Gaussian'):
            self._regressor.theta0 = self._regressor.theta_# Given correlation parameter = MLE

        print len(all_creature_data)
        self.train_regressor(all_creature_data)