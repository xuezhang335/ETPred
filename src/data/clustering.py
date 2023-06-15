# this file is from /share/home/biopharm/xuxiaobin/covid19/COVID_etpred/src/scripts/1_clustering.py

from sklearnex import patch_sklearn
patch_sklearn()

import os, sys
sys.path.append(os.path.abspath("/share/home/biopharm/xuxiaobin/covid19/COVID_etpred"))

from src.utils import utils
from src.data import cluster
from src.visualization import visualize


def clustering_strain(strains_by_month, cluster_method = 'MiniBatchKMeans', n_cluster = 3, data_files = None, train_type = None):

    trigram_vecs, _ = utils.process_month(strains_by_month, squeeze = False)

    prot_vecs = cluster.squeeze_vecs(trigram_vecs, 'sum')
    print(f'Shape: {len(prot_vecs)}x{len(prot_vecs[0])}x{len(prot_vecs[0][0])}')

    clusters_by_month = cluster.cluster_month(prot_vecs, cluster_method, n_cluster)

    clusters_by_month = cluster.link_clusters(clusters_by_month)

    strains_by_month, clusters_by_month, _ = cluster.remove_outliers(strains_by_month, clusters_by_month)
    
    if data_files != None:
        visualize.show_clusters(clusters_by_month, data_files, dims = 2, method = 'PCA', train_type = train_type)

    return strains_by_month, clusters_by_month