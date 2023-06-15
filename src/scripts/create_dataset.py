import os, sys
import pandas as pd

sys.path.append(os.path.abspath("/share/home/biopharm/xuxiaobin/covid19/COVID_etpred"))
from src.data import make_dataset
from src.features import build_features
from src.data import cluster
from src.data import clustering
import time


def main(subtype, method):
    subtype_flag, data_path = make_dataset.subtype_selection(subtype)
    
    if method == 'all':
        if subtype_flag == 0:
            file_name = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H1N1/triplet_'
        elif subtype_flag == 1:
            file_name = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H3N2/triplet_'
        elif subtype_flag == 2:
            file_name = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H5N1/triplet_'
        else:
            file_name = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/covid19/triplet_'
    elif method == 'single':
        if subtype_flag == 0:
            file_name = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H1N1_single_position/triplet_'
        elif subtype_flag == 1:
            file_name = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H3N2_single_position/triplet_'
        elif subtype_flag == 2:
            file_name = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/H5N1_single_position/triplet_'
        else:
            file_name = '/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/data/processed/covid19_single_position/triplet_'

    parameters = {
        # Relative file path to raw data sets (should contain trigram vec file)
        'data_path': data_path,

        # Year to start from
        'start_year': 2021,

        # Month to start from (useless for flu)
        'start_month': 1,

        # Year to make prediction for
        'end_year': 2023,

        # Month to make prediction for (useless for flu)
        'end_month': 1,

        # 'random' (no clustering), 'DBSCAN', 'KMeans', 'MeanShift', 'MiniBatchKMeans'
        'clustering_method': 'MiniBatchKMeans',

        # Number of clusters
        'n_cluster': 3,

        # Number of strains sampled for training
        'training_samples': 6400,

        # Number of strains sampled for validation
        'testing_samples': 1600,

        # File name to give the created data set (will be appended by clustering method and train/test)
        'file_name': file_name
    }
    
    # Epitopes sites for the H5 protein
    if subtype_flag == 0:
        epitope_a = [118, 120, 121, 122, 126, 127, 128, 129, 132, 133, 134, 135, 137, 139, 140, 141, 142, 143, 146, 147, 149, 165, 252, 253]
        epitope_b = [124, 125, 152, 153, 154, 155, 156, 157, 160, 162, 163, 183, 184, 185, 186, 187, 189, 190, 191, 193, 194, 196]
        epitope_c = [34, 35, 36, 37, 38, 40, 41, 43, 44, 45, 269, 270, 271, 272, 273, 274, 276, 277, 278, 283, 288, 292, 295, 297, 298, 302, 303, 305, 306, 307, 308, 309, 310]
        epitope_d = [89, 94, 95, 96, 113, 117, 163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174, 176, 179, 198, 200, 202, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 222, 223, 224, 225, 226, 227, 235, 237, 239, 241, 243, 244, 245]
        epitope_e = [47, 48, 50, 51, 53, 54, 56, 57, 58, 66, 68, 69, 70, 71, 72, 73, 74, 75, 78, 79, 80, 82, 83, 84, 85, 86, 102, 257, 258, 259, 260, 261, 263, 267]
        epitope_positions = epitope_a + epitope_b + epitope_c + epitope_d + epitope_e
        epitope_positions.sort()
    
    # Epitopes sites for the H3 subtype
    if subtype_flag == 1:
        epitope_a = [122, 124, 126, 130, 131, 132, 133, 135, 136, 137, 138, 140, 142, 143, 144, 145, 146, 150, 152, 168]
        epitope_b = [128, 129, 155, 156, 157, 158, 159, 160, 163, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198]
        epitope_c = [44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294, 297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312]
        epitope_d = [96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248]
        epitope_e = [57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265]
        epitope_positions = epitope_a + epitope_b + epitope_c + epitope_d + epitope_e
        epitope_positions.sort()
    
    # Epitopes sites for the H5 protein
    if subtype_flag == 2:
        epitope_positions = [36, 48, 53, 55, 56, 57, 62, 65, 71, 77, 78, 80, 81, 82, 83, 84, 86, 87, 91, 94, 115, 116, 117, 118, 119, 
                             120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 136, 138, 140, 141, 142, 143, 144, 145, 
                             149, 150, 151, 152, 153, 154, 155, 156, 157, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 171, 
                             172, 173, 174, 179, 182, 185, 186, 187, 189, 190, 191, 193, 200, 205, 206, 207, 212, 222, 226, 230, 242, 
                             244, 245, 246, 252, 256, 259, 261, 262, 263, 273, 274, 276, 278, 282]
        #epitope_positions = np.unique(epitope_positions)
        epitope_positions.sort()

    # RBD sites for the SARS -CoV-2 S protein (319-541)
    if subtype_flag == 3:
        epitope_positions = [pos for pos in range(318,541)]

    # get file path
    data_files = []
    if subtype_flag == 3:
        for year in range(parameters['start_year'], parameters['end_year'] + 1):
            for month in range(1, 13):
                if year == parameters['start_year'] and month < parameters['start_month']:
                    continue
                if year == parameters['end_year'] and month > parameters['end_month']:
                    break
                file_name = f'{year}-{month:02d}.csv'
                data_files.append(file_name)
    else:
        for year in range(parameters['start_year'], parameters['end_year'] + 1):
            file_name = f'{year}.csv'
            data_files.append(file_name)

    print(str(data_files))
    print(parameters['data_path'])

    time_0 = time.time()

    # read and split the strain sample
    strains_by_month = make_dataset.read_strains_from(data_files, parameters['data_path'])
    
    test_split = parameters['testing_samples'] / (parameters['training_samples'] + parameters['testing_samples'])
    train_strains_by_month, test_strains_by_month = make_dataset.train_test_split_strains(strains_by_month, test_split, parameters['clustering_method'])
    
    trigram_to_idx, _ = make_dataset.read_trigram_vec()

    time_1 = time.time()
    print('read and split done\ntime = {:.1f}.'.format(time_1 - time_0))

    # cluster and sample 
    train_strains_by_month, train_clusters_by_month = clustering.clustering_strain(train_strains_by_month, parameters['clustering_method'], parameters['n_cluster'], data_files, train_type=True)
    test_strains_by_month, test_clusters_by_month = clustering.clustering_strain(test_strains_by_month, parameters['clustering_method'], parameters['n_cluster'], data_files, train_type=False)
 
    train_strains_by_month = cluster.sample_from_clusters(train_strains_by_month, train_clusters_by_month, parameters['training_samples'], verbose=True)
    test_strains_by_month = cluster.sample_from_clusters(test_strains_by_month, test_clusters_by_month, parameters['testing_samples'], verbose=True)

    time_2 = time.time()
    print('cluster and sample done\ntime = {:.1f}.'.format(time_2 - time_1))

    #create dataset

    if method == 'all':
        if subtype_flag == 3:
            create_triplet_trigram_dataset(train_strains_by_month,
                                           trigram_to_idx, 
                                           epitope_positions, 
                                           file_name=(parameters['file_name'] + parameters['clustering_method']+ '_{}-{}_to_{}-{}_train'.format(parameters['start_year'], parameters['start_month'], parameters['end_year'], parameters['end_month']))
                                           )
            create_triplet_trigram_dataset(test_strains_by_month, 
                                           trigram_to_idx,
                                           epitope_positions, 
                                           file_name=(parameters['file_name'] + parameters['clustering_method']+ '_{}-{}_to_{}-{}_test'.format(parameters['start_year'], parameters['start_month'], parameters['end_year'], parameters['end_month']))
                                           )
        else:
            create_triplet_trigram_dataset(train_strains_by_month,
                                           trigram_to_idx, 
                                           epitope_positions, 
                                           file_name=(parameters['file_name'] + parameters['clustering_method'] + '_train')
                                           )
            create_triplet_trigram_dataset(test_strains_by_month, 
                                           trigram_to_idx,
                                           epitope_positions, 
                                           file_name=(parameters['file_name'] + parameters['clustering_method'] + '_test')
                                           )
    
    elif method == 'single':

        for position in epitope_positions:
            position_lst = [position]
            if subtype_flag == 3:
                create_triplet_trigram_dataset(train_strains_by_month,
                                               trigram_to_idx, 
                                               position_lst, 
                                               file_name=(parameters['file_name'] + parameters['clustering_method']+ '_{}-{}_to_{}-{}_train_{}'.format(parameters['start_year'], parameters['start_month'], parameters['end_year'], parameters['end_month'], position))
                                               )
                create_triplet_trigram_dataset(test_strains_by_month, 
                                               trigram_to_idx,
                                               position_lst, 
                                               file_name=(parameters['file_name'] + parameters['clustering_method']+ '_{}-{}_to_{}-{}_test_{}'.format(parameters['start_year'], parameters['start_month'], parameters['end_year'], parameters['end_month'], position))
                                               )
            else:
                create_triplet_trigram_dataset(train_strains_by_month,
                                               trigram_to_idx, 
                                               position_lst, 
                                               file_name=(parameters['file_name'] + parameters['clustering_method'] + f'_train_{position}')
                                               )
                create_triplet_trigram_dataset(test_strains_by_month, 
                                               trigram_to_idx,
                                               position_lst, 
                                               file_name=(parameters['file_name'] + parameters['clustering_method'] + f'_test_{position}')
                                               )

    time_3 = time.time()
    print('create dataset done\ntime = {:.1f}.'.format(time_3 - time_2))


def create_triplet_trigram_dataset(strains_by_month, trigram_to_idx, epitope_positions, file_name):
    """Creates a dataset in csv format.
    X: Time series of three overlapping trigram vectors, one example for each epitope.
    Y: 0 if epitope does not mutate, 1 if it does.
    """
    triplet_strains_by_month = build_features.make_triplet_strains(strains_by_month, epitope_positions)
    trigrams_by_month = build_features.split_to_trigrams(triplet_strains_by_month)
    trigram_idxs = build_features.map_trigrams_to_idxs(trigrams_by_month, trigram_to_idx)
    labels = build_features.make_triplet_labels(triplet_strains_by_month)

    acc, p, r, f1, mcc = build_features.get_majority_baselines(triplet_strains_by_month, labels)
    with open(file_name + '_baseline.txt', 'w') as f:
        f.write(' Accuracy:\t%.3f\n' % acc)
        f.write(' Precision:\t%.3f\n' % p)
        f.write(' Recall:\t%.3f\n' % r)
        f.write(' F1-score:\t%.3f\n' % f1)
        f.write(' Matthews CC:\t%.3f' % mcc)

    data_dict = {'y': labels}
    for month in range(len(triplet_strains_by_month) - 1):
        data_dict[str(month)] = trigram_idxs[month]

    pd.DataFrame(data_dict).to_csv(file_name + '.csv', index=False)


if __name__ == '__main__':
    print('Start to create the training dataset...')
    subtype = ['H1N1', 'H3N2', 'H5N1', 'covid19']
    sample_methods = ['all','single']

    main(subtype[3], sample_methods[1])

    print('work done')