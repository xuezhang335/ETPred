import sys, os
sys.path.append(os.path.abspath("/share/home/biopharm/xuxiaobin/covid19/COVID_etpred"))

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import math
import random
import time
import numpy as np
from math import floor
from src.visualization import visualize
from src.utils import validation
from src.data import cluster
from src.utils import utils

from scipy.cluster.hierarchy import linkage, fcluster

from src.data import make_dataset

def cluster_month(prot_vecs, method, n_cluster):
    clusters = []
    for month_prot_vecs in prot_vecs:

        if(method == 'DBSCAN'):
            min_samples = floor(len(month_prot_vecs)*0.01)
            clf = DBSCAN(eps=10, min_samples=min_samples, metric='euclidean').fit(month_prot_vecs)
            labels = clf.labels_
            centroids = NearestCentroid().fit(month_prot_vecs, labels).centroids_

        if(method == 'MeanShift'):
            clf = MeanShift().fit(month_prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_

        if(method == 'KMeans'):
            clf = KMeans(n_clusters=n_cluster, n_init=10)
            clf.fit(month_prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_
        
        # mini batch K-means
        if (method == 'MiniBatchKMeans'):
            clf = MiniBatchKMeans(n_clusters=n_cluster, n_init=10)
            clf.fit(month_prot_vecs)
            labels = clf.labels_
            centroids = clf.cluster_centers_

        unique, count = np.unique(labels, return_counts=True)

        cluster = {
            'data': month_prot_vecs,
            'labels':labels, 
            'centroids':centroids, 
            'population': dict(zip(unique, count))} 
        
        clusters.append(cluster)

    return clusters

def squeeze_vecs(trigram_vecs, type = 'join'):
    prot_vecs = []
    for month_trigram_vecs in trigram_vecs:
        month_vecs = []
        for strain_trigram_vecs in month_trigram_vecs:
            if type == 'join':
                strain_prot_vecs = np.concatenate(strain_trigram_vecs, axis=0)
            elif type == 'sum':
                strain_prot_vecs = [sum(items) for items in zip(*strain_trigram_vecs)]
            month_vecs.append(strain_prot_vecs)
        prot_vecs.append(month_vecs)
    
    return prot_vecs

def remove_outliers(data, clusters):
    removed_outliers_count = []
    for month_idx, cluster in enumerate(clusters):
        mask = cluster['labels'] != -1
        removed_outliers_count.append(len(cluster['labels']) - sum(mask))

        data[month_idx] = [row for row, m in zip(data[month_idx], mask) if m]
        clusters[month_idx]['labels'] = clusters[month_idx]['labels'][mask]
        clusters[month_idx]['data'] = np.array([row for row, m in zip(clusters[month_idx]['data'], mask) if m])
        if -1 in clusters[month_idx]['population']: del clusters[month_idx]['population'][-1]

    return data, clusters, removed_outliers_count

def evaluate_clusters(clusters):
    scores = []
    for cluster in clusters:
        score = silhouette_score(cluster['data'], cluster['labels'])
        scores.append(score)

    average = sum(scores) / float(len(scores))
    return average

def link_clusters(clusters):
    no_months = len(clusters)
    neigh = NearestNeighbors(n_neighbors=2)

    for month_idx in range(no_months): 
        if(month_idx == no_months-1): # last month doesn't link
            clusters[month_idx]['links'] = ['-'] 
            break 

        links = []
        current_centroids = clusters[month_idx]['centroids']
        next_month_centroids = clusters[month_idx+1]['centroids']
        neigh.fit(next_month_centroids)

        idxs_by_centroid = neigh.kneighbors(current_centroids, return_distance=False)

        for label in clusters[month_idx]['labels']:
            if (idxs_by_centroid[label][0] == -1): del idxs_by_centroid[label][0]
            links.append(idxs_by_centroid[label]) # centroid idx corresponds to label

        clusters[month_idx]['links'] = links

    return clusters

def sample_from_clusters(strains_by_month, clusters_by_months, sample_size, verbose=False):
    sampled_strains = [[]] * len(strains_by_month)

    # start sample from first cluster
    first_month_labels = clusters_by_months[0]['labels']
    first_month_population = clusters_by_months[0]['population']
    first_month_total = len(strains_by_month[0])

    for label_idx in first_month_population.keys(): # len(population) = no_clusters
        cluster_proportion = first_month_population[label_idx]/first_month_total
        cluster_sample_size = math.floor(sample_size*cluster_proportion)
        cluster_strains = [strains_by_month[0][i] for i, label in enumerate(first_month_labels) if label == label_idx]
        sampled_strains[0] = sampled_strains[0] + random.choices(cluster_strains, k=cluster_sample_size)

        # on last iteration sample missing
        missing_samples = sample_size - len(sampled_strains[0])
        if label_idx == list(first_month_population)[-1] and missing_samples > 0:
            print(f'Missing samples: {missing_samples}')
            sampled_strains[0] = sampled_strains[0] + random.choices(cluster_strains, k=missing_samples)


    # sample forward
    #current_cluster = label_encode([sampled_strains[0]])[0]
    current_strain = [sampled_strains[0]]
    trigram_vecs, _ = utils.process_month(current_strain, squeeze = False)
    current_cluster = cluster.squeeze_vecs(trigram_vecs, 'sum')
    current_cluster = current_cluster[0]

    for month_idx in range(1, len(clusters_by_months)):
        month_clusters = clusters_by_months[month_idx]

        if(verbose): 
            print(f'\n>>> Linking {month_idx} month')
            print(f'Clusters\n{month_clusters["population"]}')

        neigh = NearestNeighbors(n_neighbors=1, metric='hamming')
        neigh.fit(month_clusters['data'])
        neighbour_strain_idx = neigh.kneighbors(current_cluster, return_distance=True)

        nice_neighs = [idx[0] for idx in neighbour_strain_idx[1]]
        links = [month_clusters['labels'][idx] for idx in nice_neighs]

        if(verbose): 
            validation.list_summary('Neighbours', nice_neighs)
            validation.list_summary('Links', links)

        clustered_strains = {}
        for label_idx in month_clusters['population'].keys():
            clustered_strains[label_idx] = [strains_by_month[month_idx][i] for i, label in enumerate(month_clusters['labels']) if label == label_idx]

        for link in links:
            sample = random.choice(clustered_strains[link])
            sampled_strains[month_idx] = sampled_strains[month_idx] + [sample]

    return sampled_strains

