# getting start with pandas
import pandas as pd
x=pd.Series([6,3,8,6])
x=pd.Series([6,3,8,6],
			index=list("qwer"))
print(x)
print(x['w'])

age={"Tim":29, "Jim":31, "Adam":19}
y = pd.Series(age)


data = {'name': ['Tim', 'Jim', 'Pam','Sam'],
		'age' : [  29 ,  31  ,  27  ,  35 ],
		'ZIP' : ['02115','02130','67700','00100']
		}
x = pd.DataFrame(data, columns=['name', 'age', 'ZIP'],
				 index=list("qwer"))
# default index

print(x['name'])
print(x.name) # attribute notation

x.reindex(sorted(x.index)) # reordering

y = pd.Series([7,3,5,2], index=['e', 'q', 'r', 't'])

#print(x+y) # NaN


#loading data
whisky = pd.read_csv('whiskies.txt')
whisky['region'] = pd.read_csv('regions.txt')

print(whisky.head())
whisky.iloc[0:10] #rows
whisky.iloc[5:10, 0:5]
whisky.columns
flavor = whisky.iloc[:, 2:14] # flavors subset

# exploring correlations
# Pearson by default
corr_flavors = pd.DataFrame.corr(flavor)
# same as
cor = flavor.corr()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.pcolor(cor) # instead of Seaborn heatmap function
plt.colorbar()
plt.savefig('corr_flavor.pdf')

# correlation among whiskies across flavors --> transpose
corr_whisky = flavor.transpose().corr()
plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
plt.colorbar()
plt.axis('tight')
plt.savefig('corr_whisky.pdf')

# clustering whisky by flavor profile
# spectral co-clustering from scikit-learn
# --> both variables are clustered simultaneously
# using eigenvalues and eigenvectors
# reordering will create groups/clusters

from sklearn.cluster.bicluster import SpectralCoclustering
model=SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whisky)
# entries are True of False, each row identifies a cluster, from 0 to 5
# each column identifies a row in the matrix

import numpy as np
np.sum(model.rows_, axis=1)

np.sum(model.rows_, axis=0) # should be 1 group per column
model.row_labels_

# comparing correlation matrices
whisky['group'] = pd.Series(model.row_labels_, index=whisky.index)
whisky= whisky.ix[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)

correlations = whisky.iloc[:, 2:14].transpose().corr()
correlations = np.array(correlations) # now matrix reshuffled

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title('Original')
plt.axis('tight')

plt.subplot(122)
plt.pcolor(correlations)
plt.title('Rearranged')
plt.axis('tight')
plt.savefig('correlations.pdf')

import seaborn as sns
sns.clustermap(corr_whisky)
