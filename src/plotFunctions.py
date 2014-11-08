import numpy as np
import itertools
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def plotData(reduced_data, km, algorithm):
  # Step size of the mesh. Decrease to increase the quality of the VQ.
  h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

  # Plot the decision boundary. For that, we will assign a color to each
  x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
  y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Obtain labels for each point in mesh. Use last trained model.
  Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.figure(1)
  plt.clf()
  plt.imshow(Z, interpolation='nearest',
             extent=(xx.min(), xx.max(), yy.min(), yy.max()),
             cmap=plt.cm.Paired,
             aspect='auto', origin='lower')

  plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
  # Plot the centroids as a white X
  centroids = km.cluster_centers_
  plt.scatter(centroids[:, 0], centroids[:, 1],
              marker='x', s=169, linewidths=3,
              color='w', zorder=10)
  plt.title('K-means clustering on the digits dataset (' + algorithm + '-reduced data)\n'
            'Centroids are marked with white cross')
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  plt.xticks(())
  plt.yticks(())
  plt.show()

def graphVariance(reduced_data):
  # Determine your k range
  k_range = range(1,14)

  # Fit the kmeans model for each n_clusters = k
  k_means_var = [KMeans(n_clusters=k).fit(reduced_data) for k in k_range]

  # Pull out the cluster centers for each model
  centroids = [X.cluster_centers_ for X in k_means_var]

  # Calculate the Euclidean distance from 
  # each point to each cluster center
  k_euclid = [cdist(reduced_data, cent, 'euclidean') for cent in centroids]
  dist = [np.min(ke,axis=1) for ke in k_euclid]

  # Total within-cluster sum of squares
  wcss = [sum(d**2) for d in dist]

  # The total sum of squares
  tss = sum(pdist(reduced_data)**2)/reduced_data.shape[0]

  # The between-cluster sum of squares
  bss = tss - wcss

  # elbow curve
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(k_range, bss/tss*100, 'b*-')
  ax.set_ylim((0,100))
  plt.grid(True)
  plt.xlabel('n_clusters')
  plt.ylabel('Percentage of variance explained')
  plt.title('Variance Explained vs. k')
  plt.show()

def scatterplot_matrix(data, names=[], **kwargs):
    """
    Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid.
    """
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(20,20))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            # FIX #1: this needed to be changed from ...(data[x], data[y],...)
            axes[x,y].plot(data[y], data[x], **kwargs)

    # Label the diagonal subplots...
    if not names:
        names = ['x'+str(i) for i in range(numvars)]

    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    # FIX #2: if numvars is odd, the bottom right corner plot doesn't have the
    # correct axes limits, so we pull them from other axes
    if numvars%2:
        xlimits = axes[0,-1].get_xlim()
        ylimits = axes[-1,0].get_ylim()
        axes[-1,-1].set_xlim(xlimits)
        axes[-1,-1].set_ylim(ylimits)

    return fig