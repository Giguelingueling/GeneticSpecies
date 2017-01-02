import numpy as np
from sklearn import gaussian_process
from matplotlib import pyplot as plt
def f(x):
    return x * np.sin(x)
X = np.atleast_2d([-12., 0., 6., 7., 8., 9.5, 20., -20.]).T
y = f(X).ravel()
x = np.atleast_2d(np.linspace(-30, 30, 1000)).T
gp = gaussian_process.GaussianProcess(regr='linear', corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X, y)
y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)

fig = plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma2_pred,
                        (y_pred + 1.9600 * sigma2_pred)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-100, 100)
plt.legend(loc='upper left')
plt.show()