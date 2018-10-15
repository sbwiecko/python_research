import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# generating example regression data
n=100
beta_0 = 5
beta_1 = 2
np.random.seed(1)

x = 10*ss.uniform.rvs(size=n) # between 0 and 10
y = beta_0 + beta_1 * x + ss.norm.rvs(loc=0, scale=1, size=n)

plt.figure()
plt.plot(x, y, 'o', ms=5)
xx = np.array([0, 10]) # for regression line
plt.plot(xx, beta_0 + beta_1 * xx) #deterministic, no error
plt.xlabel("$x$")
plt.ylabel('$y$')

#least squares estimation
rss = []
slopes = np.arange(-10, 15, .001) # like a gris test of different values

for slope in slopes:
	rss.append(np.sum((y - beta_0 - slope * x)**2))
print(rss)

ind_min = np.argmin(rss) # selection of the value of slope with the lowest rss
print("Estimate for the slope: ", slopes[ind_min])

plt.figure()
plt.plot(slopes, rss,)
plt.xlabel("Slope")
plt.ylabel('RSS')
plt.title("Grid determination of the slope")
	   

# simple linear regression
import statsmodels.api as sm
mod = sm.OLS(y, x) # no intercept parameter, slope only
est = mod.fit()
print(est.summary())

X = sm.add_constant(x)
mod = sm.OLS(y, X) # no intercept parameter, slope only
est = mod.fit()
print(est.summary())

def compute_rss(y_estimate, y): 
  return sum(np.power(y-y_estimate, 2)) 

def estimate_y(x, b_0, b_1): 
  return b_0 + b_1 * x 

#TSS is the total sum of squares, sum of y_outcome - mean_y
#RSS is the residual sum of squares, sum of y_outcome - predicted_y
#R² = (TSS - RSS) / TSS
rss = compute_rss(estimate_y(x, beta_0, beta_1), y) 
tss = np.sum(np.power(y - np.mean(y), 2))
print(f"R² using TSS and RSS: {(tss-rss)/tss}")


# scikit-learn for linear regression
n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ss.norm.rvs(loc=0, scale=1, size=n)

X = np.stack([x_1, x_2], axis=1) # construct variable as column matrix

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X, y)

print(lm.intercept_)
print(lm.coef_)

X_0 = np.array([2, 4]) # variable with 2 predictors x1 and x2
lm.predict(X_0.reshape(1,-1))

print(lm.score(X, y)) # R² : comparision to prediction under the hood

# assessing model accuracy
# MSE = average of (yi - f(xi))²

# training and test error
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
													train_size=.5,
													random_state=1)
lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, y_train)

lm.score(X_test, y_test) # scoring using prediction under the hood


