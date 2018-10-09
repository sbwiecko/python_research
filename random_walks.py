# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:21:41 2018

@author: Sébastien
"""

# ------------
# random walks
# ------------
import matplotlib.pyplot as plt
import numpy as np

plt.clf()

delta_x = np.random.normal(0,1,(2,10000)) # random displacement
# 2 coordinates for 5 steps

# cumulative sum
# x(t) = x(0) + sum(delta * x(k))
X_0 = np.array([[0],[0]]) # 2 rows

X = np.cumsum(delta_x, axis=1) # summing over the columns
X = np.concatenate((X_0, X), axis=1) # joining on the columns

plt.plot(X[0], X[1], "r+-", markersize=2)
plt.savefig("rw.pdf")
