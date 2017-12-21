#!/usr/bin/env python3
import argparse
import sys
import os.path
import csv
from collections import namedtuple
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import numpy as np
import pylab

Clustering = namedtuple('Clustering', ['n_clusters', 'labels', 'quality', 'score'])

OUTPUT_FILE = '/home/mzieleniewska/empi/from_hpc/data/smp/control_99rms_new_reader/clustering_parameters_SWA_75uV.csv'


def clusterize(x, n_clusters):
	X = x.reshape((len(x),1))
	clusterer = KMeans(n_clusters=n_clusters)
	labels = clusterer.fit_predict(X)
	score = silhouette_score(X, labels)
	quality = clusterer.score(X)
	return Clustering(n_clusters, labels, quality, score)


def clusterize_best(x, max_n_clusters=8):
	X = x.reshape((len(x),1))
	best = None
	for n_clusters in range(2, min(len(x), max_n_clusters)+1):
		clustering = clusterize(x, n_clusters)
		if best is None or clustering.score > best.score:
			best = clustering
	if best is None:
	 	raise ValueError()
	return best


def time_shuffle(x):
	delta = x[1:] - x[:-1]
	np.random.shuffle(delta)
	x[1:] = np.cumsum(delta)


def visualize(x, clustering: Clustering):
	for i in range(clustering.n_clusters):
		pylab.plot(np.where(clustering.labels==i, x, np.nan))
	pylab.show()


def f_score(x, clustering: Clustering):
	intra_variance = 0
	for i in range(clustering.n_clusters):
		intra_variance += np.var(x[clustering.labels == i], ddof=1)
	inter_variance = np.var(x, ddof=1)
	return inter_variance / intra_variance


def main():
	parser = argparse.ArgumentParser(description='Performs clustering of sleep transients occurrences.')
	parser.add_argument('files', nargs='+', metavar='file', help='path to *.csv files')
	namespace = parser.parse_args()

	scores = []
	for file_name in namespace.files:
		df = pd.read_csv(file_name, index_col=0)
		x = df["absolute_position"].as_matrix()
		try:
			clustering_0 = clusterize_best(x)
			# visualize(x, clustering_0)

			X = x.reshape((-1, 1))
			C0 = calinski_harabaz_score(X, clustering_0.labels)

			F0 = f_score(x, clustering_0)
			# TODO parametryczny test na F

			# poniższe dodatkowo liczy coś w stylu p-wartości
			# sprawdzając na ile klastry wyszły przypadkowo
			repeats = 1
			count_C = count_F = 0
			for i in range(repeats):
				t = np.random.random(size=len(x))
				y = (max(x) - min(x)) * t + min(x)
				y = np.array(sorted(y))
				clustering = clusterize_best(y)

				Y = y.reshape((-1, 1))
				C = calinski_harabaz_score(Y, clustering.labels)
				S = silhouette_score(Y, clustering.labels)
				F = f_score(y, clustering)

				if C > C0:
					count_C += 1
				if F > F0:
					count_F += 1
			scores.append([os.path.basename(file_name), C0, F0, count_C/repeats, count_F/repeats, clustering_0.labels.max()])
		except ValueError:
			scores.append([os.path.basename(file_name), 0, 0, 1., 1., 0])

	with open(OUTPUT_FILE, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["rec_id", "calinski_harabaz", "F0", "calinski_harabaz_p", "F0_p", "nb_clustering_labels"])
		writer.writerows(scores)


if __name__ == '__main__':
	main()
	sys.exit(0)
