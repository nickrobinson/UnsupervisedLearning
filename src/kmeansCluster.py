print(__doc__)

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
import plotFunctions

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn import datasets, metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from optparse import OptionParser

from fileHelper import Wines

###############################################################################
#Command Line Parsing
# parse commandline arguments
op = OptionParser()
op.add_option("--clusters",
              dest="num_clusters", type="int",
              help="Number of clusters to use for KMeans.")

op.add_option("--minibatch",
              action="store_true", dest="minibatch", default=False,
              help="Use batch mode k-means algorithm.")

op.add_option("--max_iter",
              dest="max_iter", type="int", default=300,
              help="Number of maximum iterations for KMeans.")

op.add_option("--analysis",
              dest="analysis", type="string", default="",
              help="Analysis type to use (PCA, ICA).")

op.add_option("--n_components",
              dest="n_components", type="int", default=2,
              help="Number of component attributes to be used with ICA, PCA algorithms.")


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
# Load data
datafile = '../data/winequality-red.csv'
wines = Wines()
wines.loadData(datafile)
X, y = shuffle(wines.data, wines.target, random_state=5)
X = X.astype(np.float64)
y = y.astype(np.float64)
offset = int(X.shape[0] * 0.40)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

if opts.analysis:
  if opts.analysis == "PCA":
    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=opts.n_components)
    reduced_data = pca.fit_transform(X_train, y_train)
    km = KMeans(n_clusters=opts.num_clusters, max_iter=opts.max_iter)
    km.fit(reduced_data)
  elif opts.analysis == "ICA":
    pdb.set_trace()
    ica = FastICA(n_components=opts.n_components)
    reduced_data = ica.fit_transform(X_train)
    km = KMeans(n_clusters=opts.num_clusters, max_iter=opts.max_iter)
    km.fit(reduced_data)
  else:
    print("ERROR: Invalid analysis option!!!")
    sys.exit()
else:
    km = KMeans(n_clusters=opts.num_clusters, max_iter=opts.max_iter)
    km.fit(X_train)

labels = km.labels_

pdb.set_trace()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_train, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(y_train, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(y_train, km.labels_))
print("Silhouette Score: %0.3f" % metrics.silhouette_score(X_train, km.labels_, metric='euclidean'))

if opts.analysis and opts.n_components == 2:
  if opts.analysis == "PCA" or opts.analysis == "ICA":
    plotFunctions.plotData(reduced_data, km, opts.analysis)
    plotFunctions.graphVariance(reduced_data)
