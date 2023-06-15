import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def show_clusters(clusters, data_files, dims=2, method='TSNE', data_path='/share/home/biopharm/xuxiaobin/covid19/COVID_etpred/reports/figures_cluster/', show=False, train_type = None, n_cluster = None):
    print('visualize start')
    for month_idx, cluster in enumerate(clusters):       
        prot_vecs = cluster['data']
        labels = cluster['labels']

        if(method == 'TSNE'):
            pca_50 = PCA(n_components=1)
            pca_result_50 = pca_50.fit_transform(prot_vecs)
            reduced_data = TSNE(random_state=8, n_components=dims, perplexity=1).fit_transform(pca_result_50)
            print('tsne done.\nvisualized_dimensions = ' + str(dims))
        if(method == 'PCA'):
            pca = PCA(n_components=dims)
            reduced_data = pca.fit_transform(prot_vecs)
            print(f'Explained variance:{pca.explained_variance_ratio_}')  
            # reduced_centroids = NearestCentroid().fit(reduced_data, labels).centroids_
            print('pca done.\nvisualized_dimensions = ' + str(dims))

        if dims == 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced_data[:, 0],
                                 reduced_data[:, 1],
                                 c = labels,
                                 s = 5)
            legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
            ax.add_artist(legend1)
        elif dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data[:, 0],
                                 reduced_data[:, 1],
                                 reduced_data[:, 2],
                                 c=labels,
                                 s=10)
            legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
            ax.add_artist(legend1)
        
        if n_cluster is None:
            filename = data_path + data_files[month_idx][:-4]
        else:
            filename = data_path + data_files[month_idx][:-4] + f'_{n_cluster}'

        if train_type is None:
            filename = filename + '.png'
        elif train_type is True:
            filename = filename + '_train.png'
        elif train_type is False:
            filename = filename + '_test.png'

        print(str(filename))
        plt.savefig(filename)

        print('save done')

