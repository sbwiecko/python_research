import matplotlib.pyplot as plt

plt.plot([0,1,4,9,16])
plt.show()

import numpy as np

x=np.linspace(0,10,20)
y=x**2

x

plt.plot(x,y)

y1=x**2
y2=x**1.5

plt.plot(x, y1, "bo-", linewidth=2, markersize=10) # keyword and param blue, circle and line
plt.plot(x, y2, "gs--", linewidth=2, markersize=10) # keyword and param blue, circle and line

#-----
x=np.linspace(0,10,20)
y1=x**2
y2=x**1.5
plt.plot(x, y1, "bo-", linewidth=2, markersize=10,
         label='First') # label used in the legend
plt.plot(x, y2, "gs--", linewidth=2, markersize=10,
         label='Second')
plt.xlabel("$X$") # math formating of the axis
plt.ylabel("$Y$")
plt.axis([-.5, 10.5, -5, 105]) # [xmin, xmax, ymin, ymax]
plt.legend(loc="upper left")
plt.savefig("myplot.pdf")

#-----
# Plotting using logarithmic axes

#semilogx(), semilogy() or loglog()
#by default log10

#y=x^a transformed in loglog gives yy=a.xx
#loglog plot will give a line plot passing through (0,0)
x=np.logspace(-1,1,40)
y1=x**2
y2=x**1.5

plt.loglog(x, y1, "bo-", linewidth=2, markersize=10,
         label='First') # label used in the legend
plt.loglog(x, y2, "gs--", linewidth=2, markersize=10,
         label='Second')
plt.xlabel("$X$") # math formating of the axis
plt.ylabel("$Y$")
plt.legend(loc="upper left")
plt.savefig("myplot_loglog.pdf")

#-----
#Generating histograms

x=np.random.normal(size=1000)
plt.hist(x) # 10 bins by defaults
plt.hist(x, normed=True) # y = proportion of observations
plt.hist(x, normed=True, bins=np.linspace(-5,5, 21))

#bin = start-stop; for 20 bins we need 21 points!

""" subplot(row,col,n) or concatenated format subplot(rcn)
+-+-+-+
|1|2|3|
+-----+
|4|5|6|
+-----+
"""
x=np.random.gamma(2,3,100_000)

plt.hist(x, normed=True, cumulative=True, bins=30,
         histtype='step')

plt.figure()
plt.subplot(221)
plt.hist(x, bins=30)
plt.subplot(222)
plt.hist(x, bins=30, normed=True)
plt.subplot(223)
plt.hist(x, bins=30, normed=True, cumulative=True)
plt.subplot(224)
plt.hist(x, bins=30, normed=True, cumulative=True, histtype='step')