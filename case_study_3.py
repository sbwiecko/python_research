# -*- coding: utf-8 -*-
"""
Case Study 3 - Introduction to classification
"""
# Finding the Distance Between Two Points

def distance(p1, p2):
	"""
	Find the distance between points p1 and p2.
	"""
	# Euclidean distance
	return np.sqrt(np.sum(np.power(p2 - p1, 2)))

import numpy as np
import random
# in case of tie, randomly pick the winner


p1 = np.array([1,1])
p2 = np.array([4,4])

p2 - p1 # operation on vector


def majority_vote(votes):
	"""
	
	"""
	vote_counts = {}
	for vote in votes:
		if vote in vote_counts:
			vote_counts[vote] += 1
		else:
			vote_counts[vote] = 1
	
	winners = []
	max_count = max(vote_counts.values())
	for vote, count in vote_counts.items():
		if count == max_count:
			winners.append(vote)

	return random.choice(winners)


# Majority vote
votes = [1,2,3,1,2,3,1,2,3,1,1,1,1,3,3,3,3,3,3,1,1,3,1]
vote_counts=majority_vote(votes)

max(vote_counts)
max(vote_counts.values())
#could alos use the max(à function on array

winners = []
max_counts = max(vote_counts.values())
for vote, count in vote_counts.items():
	if count == max_counts:
		winners.append(vote)


# finally this is similar to finding the mode
import scipy.stats as ss
def majority_vote_short(votes):
	"""
	Return the most common element in votes.
	"""
	mode, count = ss.mstats.mode(votes)
	return mode # doesn't pick randomly when tie
	

##### Finding the nearest neighbors

# loop over all points
	# copute distance between point p and every other point
# sort distances and return those k points that are nearest to point p
	
	
points = np.array([[1,1], [1,2], [1,3],
				   [2,1], [2,2], [2,3],
				   [3,1], [3,2], [3,3]]) # test dataset
	
p = np.array([2.5, 2])

import matplotlib.pyplot as plt

plt.plot(points[:,0], points[:,1], 'ro')
plt.plot(p[0], p[1], 'bo')

plt.axis([.5, 3.5, .5, 3.5])

distances = np.zeros(points.shape[0])

for i in range(len(distances)):
	distances[i] = distance(p, points[i])

distances[4]

ind = np.argsort(distances)
distances[ind] # distance values sorted
distances[ind[:2]] # gives the 2 shortest distances


def find_nearest_neighbors(p, points, k=5):
	"""
	Find the k nearest neighbors of point p and return their indices.
	"""
	distances = np.zeros(points.shape[0])
	for i in range(len(distances)):
		distances[i] = distance(p, points[i])
	ind = np.argsort(distances)
	return ind[:k]

ind = find_nearest_neighbors(p, points, 2); print(points[ind])
# returns the coordinates of the 2 closest points

ind = find_nearest_neighbors(p, points, 3); print(points[ind])


def knn_predict(p, points, outcomes, k=5):
	# find k nearest neighbors
	ind = find_nearest_neighbors(p, points, k)
	# predict the class/category of p based on majority vote
	return majority_vote(outcomes[ind])
	
# allocation of category to each point
outcomes = np.array([0,0,0,0,1,1,1,1,1])

knn_predict(np.array([2.5, 2.7]), points, outcomes, k=2)
knn_predict(np.array([0.5, 1.7]), points, outcomes, k=2)


##### Generate synthetic data
ss.norm(0,1).rvs((5,2))
ss.norm(1,1).rvs((5,2))

np.concatenate((ss.norm(0,1).rvs((5,2)),
				ss.norm(1,1).rvs((5,2))), axis=0)

n=5
np.concatenate((ss.norm(0,1).rvs((n,2)), # n points with 2 coordinates
				ss.norm(1,1).rvs((n,2))), axis=0)

np.repeat(0, n)

# generation of one array of 10 elements
outcomes = np.concatenate((np.repeat(0, n),
				np.repeat(1, n)))

def generate_synthetic_data(n=50):
	"""
	Create 2 sets of points from bivariate normal distributions.
	"""
	points = np.concatenate((ss.norm(0,1).rvs((n,2)),
						  ss.norm(1,1).rvs((n,2))), axis=0)
	outcomes = np.concatenate((np.repeat(0, n),
							np.repeat(1, n)))
	
	return (points, outcomes)

n=100
points, outcomes = generate_synthetic_data(n)

plt.figure()
plt.plot(points[:n,0], points[:n,1], 'ro')
plt.plot(points[n:,0], points[n:,1], 'bo')
plt.savefig('bivariate.pdf')


##### Making a prediction grid
def make_prediction_grid(predictors, outcomes, limits, h, k):
	"""
	Classify each point on the prediction grid.
	"""
	(x_min, x_max, y_min, y_max) = limits # tuple unpacking
	xs = np.arange(x_min, x_max, h)
	ys = np.arange(y_min, y_max, h)
	xx, yy = np.meshgrid(xs, ys) # provides a matrix of nrow arrays for x
	# and ncol arrays for y

	prediction_grid = np.zeros(xx.shape, dtype=int)
	
	for i,x in enumerate(xs):
		for j,y in enumerate(ys):
			p = np.array([x,y]) # point
			prediction_grid[j,i] = knn_predict( # first index is the row = y
					p, predictors, outcomes, k)
	return (xx, yy, prediction_grid)


#### use the plot_prediction_grid.py script downloaded from the edX website

### generate data and plot the grid
	
(predictors, outcomes) = generate_synthetic_data()

k=5; filename='knn_synth_5.pdf'
limits = (-3,4,-3,4); h=.1

(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes,
limits, h, k)

plot_prediction_grid(xx, yy, prediction_grid, filename)



k=50; filename='knn_synth_50.pdf'
limits = (-3,4,-3,4); h=.1

(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes,
limits, h, k)

plot_prediction_grid(xx, yy, prediction_grid, filename)


##### Applying homemade and scikit-learn classifiers to real dataset
from sklearn import datasets
iris = datasets.load_iris()

predictors = iris.data[:, 0:2] # focus on a subset of 2 variables
outcomes = iris.target

plt.plot(predictors[outcomes==0][:,0],
		 predictors[outcomes==0][:,1], "ro")
plt.plot(predictors[outcomes==1][:,0],
		 predictors[outcomes==1][:,1], "go")
plt.plot(predictors[outcomes==2][:,0],
		 predictors[outcomes==2][:,1], "bo")
plt.savefig('iris.pdf')

### prediction grid

k=5; filename='iris_grid.pdf'
limits = (4,8,1.5,4.5); h=.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes,
limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

#######
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(predictors, outcomes)
sk_predictions = knn.predict(predictors)

##### comparison

my_predictions = np.array([knn_predict(p, predictors, outcomes, 5)
for p in predictors])

print(100 * np.mean(sk_predictions == my_predictions))
# 96% agreement between both predictions

print(100*np.mean(sk_predictions == outcomes))
print(100*np.mean(my_predictions == outcomes))
# homemade algorithm a bit better with 84.7% correct preodictions vs 83.3%
####################################################""