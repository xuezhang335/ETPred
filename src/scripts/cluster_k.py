from sklearnex import patch_sklearn
patch_sklearn()

import os, sys
sys.path.append(os.path.abspath("/share/home/biopharm/xuxiaobin/covid19/COVID_etpred"))

from src.utils import utils
from src.data import cluster
from src.visualization import visualize
from src.data import make_dataset
import time

subtype = 'covid19'
_, data_path = make_dataset.subtype_selection(subtype)

start_year = 2020
start_month = 3
end_year = 2023
end_month = 2

clustering_method = 'MiniBatchKMeans'
n_clusters = [2,3,4,5]

reduction_method = 'PCA'
visualized_dimensions = 2 

# get file path
data_files = []
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        if year == start_year and month < start_month:
            continue
        if year == end_year and month > end_month:
            break
        file_name = f'{year}-{month:02d}.csv'
        data_files.append(file_name)

print(data_path)
print(str(data_files))
print(str(clustering_method))

start_time = time.time()

# read and trigram
strains_by_month = make_dataset.read_strains_from(data_files, data_path)
trigram_vecs, _ = utils.process_month(strains_by_month, squeeze = False)

time_1 = time.time()
print('time for trigram: {:.1f}.'.format(time_1 - start_time)) 

# protvec
prot_vecs = cluster.squeeze_vecs(trigram_vecs, 'sum')

print(f'Shape: {len(prot_vecs)}x{len(prot_vecs[0])}x{len(prot_vecs[0][0])}')
time_2 = time.time()
print('time for vec: {:.1f}.'.format(time_2 - time_1)) 

# cluster
for n_cluster in n_clusters:
    clusters_by_month = cluster.cluster_month(prot_vecs, clustering_method, n_cluster)

    average = cluster.evaluate_clusters(clusters_by_month)
    print(f'Average variance {n_cluster}: {average}')

    clusters_by_month = cluster.link_clusters(clusters_by_month)
    time_3 = time.time()
    print('time for cluster and link: {:.1f}.'.format(time_3 - time_2)) 

    # remove
    strains_by_month, clusters_by_month, n_remove = cluster.remove_outliers(strains_by_month, clusters_by_month)
    time_4 = time.time()
    print('number of remove = ' + str(n_remove))
    print('time for remove: {:.1f}.'.format(time_4 - time_3)) 

    # visualize
    visualize.show_clusters(clusters_by_month, data_files, dims = visualized_dimensions, method = reduction_method, n_cluster=n_cluster)
    time_5 = time.time()
    print('time for show cluster: {:.1f}.'.format(time_5 - time_4))
    print('time for all: {:.1f}.'.format(time_5 - start_time)) 
    
    
print('done')