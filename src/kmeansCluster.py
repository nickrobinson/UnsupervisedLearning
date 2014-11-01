print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
import plotFunctions

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn import datasets, metrics
from sklearn.decomposition import PCA

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
              dest="analysis", type="str", default="",
              help="Analysis type to use (PCA, ICA).")

op.add_option("--n_components",
              dest="n_components", type="int", default=2,
              help="Number of component attributes to be used with ICA, PCA algorithms.")

print(__doc__)
op.print_help()

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
offset = int(X.shape[0] * 0.20)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

if opts.analysis:
  if opts.analysis == "PCA":
    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=opts.n_components)
    reduced_data = pca.fit_transform(X_train)
    km = KMeans(n_clusters=opts.num_clusters, max_iter=opts.max_iter)
    km.fit(reduced_data)
else:
    km = KMeans(n_clusters=opts.num_clusters, max_iter=opts.max_iter)
    km.fit(pca)

labels = km.labels_

print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_train, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(y_train, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(y_train, km.labels_))

if opts.analysis:
  if opts.analysis == "PCA":
    plotFunctions.plotPCAData(reduced_data, km)
