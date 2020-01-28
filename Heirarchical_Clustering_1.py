import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values

# using the dendogram to find the number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('customers')
plt.ylabel('Eculidien distances')
plt.show()

# Fitting hierarchical clustering to the mall dataset
from  sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5, linkage='ward', affinity='euclidean')
y_hc = hc.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 10, c = 'red', label = 'Careful')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 10, c = 'blue', label = 'Standard')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 10, c = 'green', label = 'Target')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 10, c = 'cyan', label = 'Careless')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 10, c = 'magenta', label = 'Sensible')
plt.title("clusters of clients")
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()
