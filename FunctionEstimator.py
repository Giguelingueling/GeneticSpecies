from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from scipy.stats import norm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class FunctionEstimator(object):

    def __init__(self, get_EI=False):
        self._regressor = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=4)
        self._get_EI = get_EI

    def set_EI_bool(self, EI_bool):
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
        prediction, std = self.get_estimation_fitness_value(position)
        if std == 0.0:
            expected_improvement = 0.0
        else:
            gamma = (best_fitness-prediction)/std
            expected_improvement = std * gamma * (norm.cdf(gamma) + norm.pdf(gamma))
            expected_improvement *= -1  # We multiply by -1 to have a minimisation problem
        return expected_improvement, std

    def train_regressor(self, X, y, choose_kernel=False):
        print "UPDATING REGRESSOR"
        # self._regressor.theta0 = self._regressor.kernel_.theta  # Given correlation parameter = MLE
        print len(y)
        choose_kernel = False
        if choose_kernel is False:
            self._regressor.fit(np.array(X), np.array(y))
        else:
            print "Optimizing Kernel..."
            prediction_Matern_five_two = []
            prediction_rationalQuadratic = []
            prediction_AbsoluteExponentialKernel = []
            prediction_cubic = []

            # Create the GP with the associated kernel and add a small white noise to protect from numerical instability
            matern_five_two = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=4)
            absoluteExponentialKernel = GaussianProcessRegressor(kernel=Matern(nu=0.5), n_restarts_optimizer=4)
            cubic = GaussianProcessRegressor(kernel=RBF()**3 + RBF()**2 + RBF(), n_restarts_optimizer=4)


            # Do cross-validation with leave one out
            kf = KFold(len(y))
            for train_index, test_index in kf.split(X):
                train_cv_X = [X[index] for index in train_index]
                train_cv_y = [y[index] for index in train_index]

                matern_five_two.fit(train_cv_X, train_cv_y)
                prediction_Matern_five_two.append(matern_five_two.predict([X[test_index]])[0])

                absoluteExponentialKernel.fit(train_cv_X, train_cv_y)
                prediction_AbsoluteExponentialKernel.append(absoluteExponentialKernel.predict([X[test_index]])[0])

                cubic.fit(train_cv_X, train_cv_y)
                prediction_cubic.append(cubic.predict([X[test_index]])[0])

            # Choose best kernel based on the mean square root error.
            mse_matern_five_two = mean_squared_error(y, prediction_Matern_five_two)
            mse_ExpKernel = mean_squared_error(y, prediction_AbsoluteExponentialKernel)
            mse_cubic = mean_squared_error(y, prediction_cubic)

            list_mse = [mse_matern_five_two, mse_ExpKernel, mse_cubic]
            print list_mse
            min_mse = min(list_mse)
            print "Optimization Kernel Done"
            if min_mse == mse_matern_five_two:
                print "Selected Matern 2.5"
                self._regressor = matern_five_two
            elif min_mse == mse_ExpKernel:
                self._regressor = absoluteExponentialKernel
                print "Selected Exp Kernel (Matern 0.5)"
            else:
                print "Selected Cubic Kernel"
                self._regressor = cubic
            self._regressor.fit(np.array(X), np.array(y))
