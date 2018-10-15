"""
The goal is to classify 2 sets of data according to the distribution
of X1 and X2 in the 2D space, e.g. data points on the left part
of the y-axis will be classified as class 1, and on the right part
as class 2. The points are normally distributed according to the distance h
to the origin.
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
%matplotlib notebook

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# generating example classification data
h = 1
sd = 1
n = 50

def gen_data(n, h, sd1, sd2):
	x1 = ss.norm.rvs(-h, sd1, n)
	y1 = ss.norm.rvs(0, sd1, n)
	x2 = ss.norm.rvs(h, sd2, n) # right part
	y2 = ss.norm.rvs(0, sd2, n)
	
	return x1, y1, x2, y2

(x1, y1, x2, y2) = gen_data(1000, 1.5, 1, 1.5)

def plot_data(x1, y1, x2, y2):
	plt.figure()
	plt.plot(x1, y2, "o", ms=2)
	plt.plot(x2, y2, "o", ms=2)
	plt.xlabel("$X_1$")
	plt.ylabel("$X_2$")
	plt.show()
	
plot_data(x1, y1, x2, y2)


#logistic regression
#log(p_x / (1-p_x) = beta_0 + beta_1.X1)

def prob_to_odd(p):
	"""Converts probability to odds"""
	if p<=0 or p>=1:
		print("Probabilities must be between 0 and 1.")
	return p/(1-p)

#what is the odds that a given data point belongs to class_2 with
#p(class_1)=0.2
	
print(prob_to_odd(1 - .2))


#logistic regression in code
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression() # classifier
# y is a vector with outcome either 1 or 2
# X is a matrix with n obs with values=covariates (prodictors)
# stack x1 with x2, same for x2 and y2, and stack 1st block on top of 2nd
np.vstack((x1, y1)).shape
np.vstack((x1, y1)).T.shape # ths is the good shape, 2 cols
np.hstack((x1, y1)).shape # concatenate by the right, 1 row...
np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1))).shape
np.vstack((x1, y1)).T is np.vstack((x1, y1)).reshape(-1, 2)
np.vstack((x1, y1)).T is np.hstack((x1.reshape(-1, 1), y1.reshape(-1, 1)))

X = np.vstack(
		(np.hstack(
				(x1.reshape(-1, 1), y1.reshape(-1, 1))
				), np.hstack((x2.reshape(-1, 1), y2.reshape(-1, 1))))
			   )
X.shape

n = 1000
y = np.hstack((np.repeat(1, n), np.repeat(2, n))) # building the y vector
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5,
													random_state=1)
y_train.shape

clf.fit(X_train, y_train)
clf.score(X_test, y_test) # gives .894
clf.predict_proba(np.array([-2, 0]).reshape(1, -1))
# array([[0.96701634, 0.03298366]]) proba for the 2 classes
clf.predict(np.array([-2, 0]).reshape(1, -1))
# array([1]) class_1
clf.predict(np.array([4, -2]).reshape(1, -1))
# array([2]) class_2, see how the y vector was build
clf.predict_proba(np.array([.5, -.5]).reshape(1, -1))


#computing predictive probabilities accross the grid
# prediction for each point of the meshgrid 

# matrix to vector using.ravel(), and transpose after stacking
def plot_probs(ax, clf, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1))
    Z = probs[:,class_no]
    Z = Z.reshape(xx1.shape)
    CS = ax.contourf(xx1, xx2, Z)
    cbar = plt.colorbar(CS)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
	
plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0)
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1)
plt.title("Pred. prob for class 2")
plt.savefig('grid.pdf');
# with up to 2 predictors, the linear classification might work good enough
