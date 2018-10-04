# -*- coding: utf-8 -*-
"""
In this case study, we will analyze a dataset consisting of an assortment 
of wines classified as "high quality" and "low quality" and will use the 
k-Nearest Neighbors classifier to determine whether or not other information 
about the wine helps us correctly predict whether a new wine will be of high 
quality.
Our first step is to import the dataset.
"""
import pandas as pd
#data = pd.read_csv("https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv")

from sklearn import datasets
wine=datasets.load_wine()
data=pd.DataFrame(data=wine['data'], columns=wine['feature_names'])
data['Wine'] = wine['target']+1"

"""
Print the first 5 rows of data using the head() method.
The dataset remains stored as data. Two columns in data are is_red and 
color, which are redundant. Drop color from the dataset, and save the new 
dataset as numeric_data.
"""

print(data.head())
#numeric_data = data.drop('color')
numeric_data = data.copy()

"""
We want to ensure that each variable contributes equally to the kNN 
classifier, so we will need to scale the data by subtracting the mean of 
each column and dividing each column by its standard deviation. Then, we 
will use principal components to take a linear snapshot of the data from 
several different angles, with each snapshot ordered by how well it aligns 
with variation in the data. In this exercise, we will scale the numeric 
data and extract the first two principal components.
Scale the data using the sklearn.preprocessing function scale() on 
numeric_data.
Convert this to a pandas dataframe, and store as numeric_data.
Include the numeric variable names using the parameter columns = 
numeric_data.columns.
Use the sklearn.decomposition module PCA(), and store this as pca.
Use the fit_transform() function to extract the first two principal 
components from the data, and store this as principal_components.
"""

import sklearn.preprocessing
scaled_data = sklearn.preprocessing.scale(numeric_data)
numeric_data = pd.DataFrame(data=scaled_data,
							columns=numeric_data.columns)

import sklearn.decomposition
pca = sklearn.decomposition.PCA()
principal_components = pca.fit_transform(numeric_data)

"""
In this exercise, we will plot the first two principal components of the 
covariates in the dataset. The high and low quality wines will be colored 
using red and blue.
The first two principal components can be accessed using 
principal_components[:,0] and principal_components[:,1]. 
Store these as x and y respectively, and plot the first two principal 
components. Consider: how well are the two groups of wines separated by 
the first two principal components?
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['Wine'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.show()


"""
In this exercise, we will create a function that calculates the accuracy 
between predictions and outcomes.
Create a function accuracy(predictions, outcomes) that takes two lists of 
the same size as arguments and returns a single number, which is the 
percentage of elements that are equal for the two lists.
Use accuracy to compare the percentage of similar elements in 
x = np.array([1,2,3]) and y = np.array([1,2,4]). Print your answer.
"""
def accuracy(predictions, outcomes):
	return 100 * np.sum(predictions == outcomes) / len(outcomes)

x=np.array([1,2,3]); y=np.array([1,2,4])
print(accuracy(x, y))


"""
The dataset remains stored as data. Because most wines in the dataset are 
classified as low quality, one very simple classification rule is to 
predict that all wines are of low quality. In this exercise, we determine 
the accuracy of this simple rule. The accuracy() function preloaded into 
memory as defined in Exercise 5.
"""
print(accuracy(0, data['Nonflavanoid.phenols']))


"""
In this exercise, we will use the kNN classifier from scikit-learn to 
predict the quality of wines in our dataset.
Use knn.predict(numeric_data) to predict which wines are high and low 
quality and store the result as library_predictions.
Use accuracy to find the accuracy of your predictions, using 
library_predictions as the first argument and data["high_quality"] as 
the second argument. Print your answer. Is this prediction better 
than the simple classifier in Exercise 6?
"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['Wine'])
library_predictions = knn.predict(numeric_data)
print(accuracy(library_predictions, data['Wine']))


"""
Unlike the scikit-learn function, our homemade kNN classifier does not 
take any shortcuts in calculating which neighbors are closest to each 
observation, so it is likely too slow to carry out on the whole dataset. 
In this exercise, we will select a subset of our data to use in our 
homemade kNN classifier.
To circumvent this, fix the random generator using random.seed(123), 
and select 10 rows from the dataset using random.sample(range(n_rows), 10). 
Store this selection as selection.
"""
import random
n_rows = data.shape[0] # same as len(data)
random.seed(123)
selection = random.sample(range(n_rows), 10)


"""
We are now ready to use our homemade kNN classifier and compare the 
accuracy of our results to the baseline. The sample of 10 row indices are 
stored as selection from the previous exercise.
For each predictor p in predictors[selection], use knn_predict(p, 
predictors[training_indices,:], outcomes, k=5) to predict the quality of 
each wine in the prediction set, and store these predictions as a np.array 
called my_predictions. Note that knn_predict is already defined as in the 
Case 3 videos. Using the accuracy function, compare these results to the 
selected rows from the high_quality variable in data using my_predictions 
as the first argument and data.high_quality[selection] as the second 
argument. Store these results as percentage. Print your answer.
"""
predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["Wine"])

my_predictions =  np.array(
		[knn_predict(p, predictors[training_indices,:], outcomes, 5
			   ) for p in predictors[selection]])
percentage = accuracy(my_predictions, data['Wine'][selection])
print(percentage)





