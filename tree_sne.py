from fast_tsne import fast_tsne # this is defined in PYTHONPATH
#from sklearn.cluster import DBSCAN, OPTICS
from sklearn.mixture import BayesianGaussianMixture
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import mode, binned_statistic, ttest_ind
SEED = 37
np.random.seed(SEED)

from datasets import sbm

from display_tree import *

from load_data import *

class TreeSNE():
    # grow a tree by initializing (1D) embedding
    # then modifying embedding with smaller and smaller df
    # returns matrix with points and different 1D embedding
    # locations
    # later, return matrix containing cluster level assignments
    def __init__(self, init_df = 1, df_ratio = .9, rand_state = SEED, ignore_later_exag = True, map_dims = 1, perp = None, late_exag_coeff = 4, dynamic_perp = True, init_with_pca = True, max_iter = 5000):
        self.init_df = init_df
        self.df_ratio = df_ratio
        self.rand_state = rand_state
        # assert ignore_later_exag is False, "ignore_later_exag not yet supported"
        self.ignore_later_exag = ignore_later_exag
        self.map_dims = map_dims
        self.perp = perp
        self.late_exag_coeff = late_exag_coeff
        self.max_iter = max_iter

        self.dynamic_perp = dynamic_perp
        if dynamic_perp:
            self.curr_perp = self.perp

        self.init_with_pca = init_with_pca

        self.curr_df = self.init_df

    def _grow_tree_once(self, X, init_embed):
        self.curr_df *= self.df_ratio
        if self.dynamic_perp:
            self.curr_perp = self.curr_perp ** self.df_ratio
            # self.curr_perp = 1 + self.curr_perp * self.df_ratio

        new_embed = fast_tsne(
            X,
            map_dims = self.map_dims,
            perplexity = self.perp if not self.dynamic_perp else self.curr_perp,
            df = self.curr_df, 
            initialization = init_embed, 
            seed = self.rand_state, # I don't think this is necessary ..
            # load_affinities = "load",
            stop_early_exag_iter = 0 if self.ignore_later_exag else 250, # 250 is default value in library,
            start_late_exag_iter = 0 if self.late_exag_coeff != -1 else -1,
            late_exag_coeff = self.late_exag_coeff,
            learning_rate = X.shape[0] / 10,
            knn_algo = 'vp-tree',
            search_k = 1,
            max_iter = self.max_iter
        )
        # print(np.unique(DBSCAN().fit_predict(new_embed)).shape)
        # self._get_tsne_clusters_via_parallel_shrinkage(X, new_embed)

        # print("number of points changed: %d"%(sum(1-np.isclose(np.sort(new_embed, axis = 0), np.sort(init_embed, axis = 0)))))

        return new_embed

    def _grow_tree(self, X, n_layers = 64):
        if self.perp is None:
            self.perp = X.shape[0] ** .5
            self.curr_perp = self.perp

        if self.init_with_pca:
            init_embed = PCA(self.map_dims).fit_transform(X)

        new_embed = fast_tsne(
            X,
            map_dims = self.map_dims, 
            perplexity = self.perp,
            df = self.init_df,
            initialization = None if not self.init_with_pca else init_embed,
            seed = self.rand_state,
            load_affinities = "save",
            start_late_exag_iter = 0 if self.late_exag_coeff != -1 else -1,
            late_exag_coeff = self.late_exag_coeff,
            learning_rate = X.shape[0] / 10,
            knn_algo = 'vp-tree',
            search_k = 1,
            max_iter = self.max_iter
        )

        embeddings = [new_embed]
        for i in range(n_layers - 1):
            new_embed = self._grow_tree_once(X, new_embed)
            embeddings.append(new_embed)

        # for new_embed in embeddings:
        #     # print(np.unique(DBSCAN(self._get_dbscan_epsilon(new_embed)).fit_predict(new_embed)).shape)
        #     # print(np.unique(OPTICS(max_eps = self._get_dbscan_epsilon(new_embed)).fit_predict(new_embed)).shape)
        # #     print(np.unique(BayesianGaussianMixture(50).fit_predict(new_embed)).shape)
        #     self._get_pop_off_clusters(new_embed)
            # self._get_tsne_clusters_via_shrinkage(X, new_embed)

        embeddings = np.array(embeddings).reshape(X.shape[0], len(embeddings), -1)

        return embeddings

    def _get_tsne_clusters_via_shrinkage(self, X, init_embed, max_iter = 50):
        # get pairwise distances between points that are next to each other
        # run tSNE for a few more steps with slightly smaller alpha
        # while keeping perplexity constant so not selecting for breaking clusters apart
        # get pairwise sequential distances again
        # look for biggest increase in distances, split dataset on that
        # look for biggest increase on either side, split dataset 
        # recurse
        # eventually won't split any more because all points will get closer together
        # if they're in the same cluster 
        # THAT IS THE THEORY
        init_embed = init_embed.reshape(-1)
        sorted_init_embed = np.sort(init_embed)
        sorted_init_embed /= sorted_init_embed[-1] - sorted_init_embed[0]
        print(sorted_init_embed[:10])
        print(sorted_init_embed.shape)
        init_seq_dists = sorted_init_embed[1:] - sorted_init_embed[:-1]
        # init_seq_dists /= init
        new_embed = fast_tsne(
            X,
            map_dims = self.map_dims,
            perplexity = self.perp,#self.perp if not self.dynamic_perp else self.curr_perp,
            df = self.curr_df * .01, 
            initialization = init_embed, 
            seed = self.rand_state, # I don't think this is necessary ..
            # load_affinities = "load",
            stop_early_exag_iter = 0 if self.ignore_later_exag else 250, # 250 is default value in library,
            start_late_exag_iter = 0 if self.late_exag_coeff != -1 else -1,
            late_exag_coeff = self.late_exag_coeff,
            learning_rate = X.shape[0] / 10,
            knn_algo = 'vp-tree',
            search_k = 1,
            max_iter = max_iter,
        )
        new_embed = new_embed.reshape(-1)
        sorted_new_embed = np.sort(new_embed)
        sorted_new_embed /= sorted_new_embed[-1] - sorted_new_embed[0]
        new_seq_dists = sorted_new_embed[1:] - sorted_new_embed[:-1]
        diffs = new_seq_dists - init_seq_dists
        print(np.sort(diffs)[-50:])
        print(np.sort(diffs)[:50])
        print(diffs)
        print(np.argmax(diffs))
        n_clusters = np.sum(diffs > 0) + 1
        print(n_clusters)

    def _get_tsne_clusters_via_parallel_shrinkage(self, X, init_embed, max_iter = 500):
        # let the embedding evolve with two different df parameters
        # and then look at distance comparison thing
        # idea is that embedding is still evolving so can't just see how it changes
        # as it moves forward, need to compare two different possible ways of moving
        # forward
        # basically see how changing a param effects the embedding changes
        new_embed = fast_tsne(
            X,
            map_dims = self.map_dims,
            perplexity = self.perp,# if not self.dynamic_perp else self.curr_perp,
            df = self.curr_df ** 2, 
            initialization = init_embed, 
            seed = self.rand_state, # I don't think this is necessary ..
            # load_affinities = "load",
            stop_early_exag_iter = 0 if self.ignore_later_exag else 250, # 250 is default value in library,
            start_late_exag_iter = 0 if self.late_exag_coeff != -1 else -1,
            late_exag_coeff = self.late_exag_coeff,
            learning_rate = X.shape[0] / 10,
            knn_algo = 'vp-tree',
            search_k = 1,
            max_iter = max_iter,
        )
        shrunk_embed = fast_tsne(
            X,
            map_dims = self.map_dims,
            perplexity = self.perp,# if not self.dynamic_perp else self.curr_perp,
            df = self.curr_df ** .5,#self.curr_df * .01, 
            initialization = init_embed, 
            seed = self.rand_state, # I don't think this is necessary ..
            # load_affinities = "load",
            stop_early_exag_iter = 0 if self.ignore_later_exag else 250, # 250 is default value in library,
            start_late_exag_iter = 0 if self.late_exag_coeff != -1 else -1,
            late_exag_coeff = self.late_exag_coeff,
            learning_rate = X.shape[0] / 10,
            knn_algo = 'vp-tree',
            search_k = 1,
            max_iter = max_iter,
        )
        sorted_new_embed = np.sort(new_embed.reshape(-1))
        sorted_new_embed /= sorted_new_embed[-1] - sorted_new_embed[0]
        new_dists = sorted_new_embed[1:] - sorted_new_embed[:-1]
        sorted_shrunk_embed = np.sort(shrunk_embed.reshape(-1))
        sorted_shrunk_embed /= sorted_shrunk_embed[-1] - sorted_shrunk_embed[0]
        shrunk_dists = sorted_shrunk_embed[1:] - sorted_shrunk_embed[:-1]
        diffs = shrunk_dists - new_dists
        n_clusters = np.sum(diffs < 0) + 1
        print(n_clusters)

    def _get_tsne_clusters_via_pop_off(self, X, df):
        # keep popping off left most cluster until no points left to embed
        clusters = []
        total_points = len(X)
        mask = np.ones((total_points), dtype = bool)
        clustered_points = 0

        while clustered_points < total_points:
            embedding = fast_tsne(
                X[mask],
                map_dims = self.map_dims,
                perplexity = self.perp,
                df = df,
                seed = self.rand_state,
                start_late_exag_iter = 0 if self.late_exag_coeff != -1 else -1,
                late_exag_coeff = self.late_exag_coeff,
                learning_rate = X.shape[0] / 10,
                knn_algo = 'vp-tree',
                search_k = 1,
            )
            inner_inds = np.argsort(embedding, axis = 0)
            cluster = self._pop_off_via_distance_t_score(embedding[inner_inds])
            cluster_inner_inds = inner_inds[cluster]
            # print(cluster)
            outer_inds = np.where(mask == True)[0][cluster_inner_inds] # something wrong with this line..
            clusters.append(outer_inds)
            mask[outer_inds] = False
            clustered_points += len(outer_inds)
            # print(clustered_points)

            # print(len(cluster))

        print("found %d clusters"%len(clusters))

        labels = self._revert_to_cluster_labels(clusters)

        return labels

    def _revert_to_cluster_labels(self, clusters):
        labels = -np.ones((sum(len(cluster) for cluster in clusters)))

        for i, cluster in enumerate(clusters):
            for point in cluster:
                labels[point] = i

        return labels


    # TODO:
    # this is suppsed to reembed each time then pop off again
    # right now it's just continuously popping off

    def _get_pop_off_clusters(self, embedding):
        # pop off edge clusters until remaining is stable cluster
        # or no points left

        # print(embedding.shape)
        # print(np.sort(embedding)[:20])

        inds = np.argsort(embedding, axis = 0)
        # print(inds)
        clusters = []
        total_points = embedding.shape[0]
        clustered_points = 0
        mask = np.ones(embedding.shape, dtype = bool)

        # print(embedding[inds][:10])
        # print(np.min(embedding))

        while clustered_points < total_points + 1:
            cluster = inds[self._pop_off_via_distance_t_score(embedding[inds][clustered_points:])]
            # print(cluster)
            print(len(cluster))
            clusters.append(inds[cluster])
            # print(len(cluster))
            clustered_points += len(cluster)
            # mask[cluster] = False
            # print(sum(mask == True))

        print(len(clusters))
        clusters = np.array(clusters)

        return clusters

    def _pop_off_via_probability_locations(self, embedding, dev = 2):
        # start at edge of cluster, add points until next point is more
        # than dev standard deviations away in terms of embedding
        # print(embedding[:10])
        cluster = [embedding[0]]
        inds = [0]

        for ind, point in enumerate(embedding[1:]):
            std = np.std(cluster)
            print(std)
            mean = np.mean(cluster)
            print(mean)
            this_dev = (point - mean) / std
            print(this_dev)
            if this_dev > dev:
                break
            else:
                cluster.append(point)
                # print(cluster)
                inds.append(ind + 1)

        # print(len(inds))

        return inds

    def _pop_off_via_cluster_diameter(self, embedding, ratio = 10, die_down = 100):
        # pop off when distance to next point is ratio times
        # the total cluster radius as seen so far
        # assumes first two are in the same cluster
        # print(embedding.shape)

        cluster = [embedding[0], embedding[1]]
        inds = [0, 1]
        diameter = embedding[1] - embedding[0]

        for ind, point in enumerate(embedding[2:]):
            new_dist = point - cluster[-1]
            if new_dist > (ratio + die_down/(ind + 2)) * diameter:
                print(new_dist)
                print(diameter)
                break
            else:
                cluster.append(point)
                inds.append(ind + 2)

        # print(len(inds))
        # print(cluster)

        return inds

    def _pop_off_via_distance_t_score(self, embedding, t_thresh = 10):
        # add point to cluster if the distance to the next point 
        # when compared to the sequential distances from before
        # has a t score lower than the threshold

        dists = embedding[1:] - embedding[:-1]
        print(dists)

        for i in range(1, dists.shape[0] - 1):
            # t, p = ttest_ind(dists[:i + 1], dists[i + 1])
            # print(t)
            sample_mean = np.mean(dists[:i + 1])
            sample_std = np.std(dists[:i + 1])
            t = np.abs((dists[i + 1] - sample_mean) / (sample_std))
            # print(t)
            # print(i)
            if t > t_thresh:
                break

        print(i)

        return np.arange(i + 1)


    # def _pop_off_via_probability(self, embedding, p = .05):
    #     # start at edge of embedding, get dists between edge points,
    #     # if the next dist is statistically significantly different from
    #     # previous dists then that's the edge of the cluster
    #     # assume distributed via Gaussian. t might be better but not sure
    #     # ac
    #     inds = np.argsort(embedding)

    # def _pop_off_via_iterative_gmm(self, embedding):
    #        # NOTE THIS HAS A SORTING ISSUE NOW

    #     # sort via embedding
    #     # start at edge of embedding
    #     # add points to cluster list
    #     # run gmm on cluster points
    #     # once gmm makes two clusters, break
    #     # assumes first two are in same cluster
    #     inds = np.argsort(embedding)
    #     print(embedding[inds][:200])
    #     cluster = list(inds[:2])

    #     for ind in inds[2:]:
    #         cluster.append(ind)
    #         # print(cluster)
    #         # assignments = BayesianGaussianMixture(2).fit_predict(np.array(cluster).reshape(-1, 1))
    #         assignments = BayesianGaussianMixture(2).fit_predict(embedding[cluster].reshape(-1, 1))
    #         if np.unique(assignments).shape[0] == 2:
    #             return cluster[:-1]

    #     # this is already stable
    #     return cluster

    def _get_dbscan_epsilon(self, embedding, bins = 20):
        # calculate theoretical optial DBSCAN epsilon param
        # not sure what this should be but think can get a good estimate
        # TODO: play with this a lot
        # for now taking percentile cutoff of pairwise distances
        # where percentile is log n / n
        dists = pdist(embedding)
        # print(dists[10, 10])
        # print(dists.shape)
        # p = 100 * np.log(embedding.shape[0]) / embedding.shape[0]
        # # print(p)
        # eps = np.percentile(dists, p)
        # binned = (bins * dists / np.max(dists)).astype(int)
        # eps = mode(binned, axis = None)[0]
        eps = binned_statistic()
        print(eps)

        return eps

    def fit(self, *args, **kwargs):
        # just wraps self._grow_tree
        return self._grow_tree(*args, **kwargs)

    # def display_tree(self, embeddings, true_labels = None):
    #     assert embeddings.shape[-1] == 1, "only 1D embedding supported right now"
    #     embeddings = embeddings[true_labels == 4]
    #     # embeddings = embeddings.squeeze()
    #     # x, y = zip(*([item, i] for i, sublist in enumerate(embeddings.T) for item in sublist))
    #     x = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1])
    #     # y = np.arange(x.shape[0]) % embeddings.shape[1]
    #     y = np.repeat(np.arange(embeddings.shape[1]), embeddings.shape[0])

    #     plt.figure()
    #     plt.hist(embeddings[-1])
    #     plt.show()

    #     # print("this isn't finished yet")
    #     plt.figure()
    #     if true_labels is None:
    #         plt.scatter(x, y)
    #     else:
    #         large_labels = np.tile(true_labels, embeddings.shape[1])
    #         plt.scatter(x, y, alpha = .01)#, c = large_labels)
    #     # plt.colorbar()
    #     plt.show()


#TODO: increase repulsion later, late exageration
# IDEA: pop off edge cluster from embedding, rerun
# iteratively pop off edge till it's all one cluster
# to determine edge cluster, start at edgemost point
# get distance to next point
# keep moving until next distance is greater than some
# parameter times the largest distance so far?
# or some parameter times the total seen cluster diameter?
# or param_1+param_2/n times cluster diamater where n is
# number of points seen so far?
# or DBSCAN where epsilon is dist between first two points?
# or where next dist is statistically significantly
# different from the dists seen so far, with param as cutoff?
# or statistically significant difference in embedding?
# idea is that popping off captures what t-SNE thinks is 
# the most different from the rest


if __name__ == "__main__":
    # legends = ['Rod BC',
    #      'Muller Glia',
    #      'BC1A',
    #      'BC1B',
    #      'BC2',
    #      'BC3A',
    #      'BC3B',
    #      'BC4',
    #      'BC5A',
    #      'BC5B',
    #      'BC5C',
    #      'BC5D',
    #      'BC6',
    #      'BC7',
    #      'BC8/9_1',
    #      'BC8/9_2',
    #      'Amacrine_1',
    #      'Amacrine_2',
    #      'Rod PR',
    #      'Cone PR'
    # ]
    # X = np.load("shekhar_data.npy")
    # print(X.shape)
    # labels = np.load("shekhar_labels.npy")
    # # subset = np.random.choice(X.shape[0], 5000, replace = False)
    # dim_reduction = PCA(100) # 100 is good 
    # # dim_reduction = SpectralEmbedding(100)
    # X = dim_reduction.fit_transform(X)
    # # print(dim_reduction.explained_variance_ratio_)
    # print(X.shape)
    # X = np.load("shekhar_data.npy")
    # labels = np.load("shekhar_labels.npy")
    # subset = np.random.choice(X.shape[0], 5000, replace = False)
    # dim_reduction = PCA(100)
    # X = dim_reduction.fit_transform(X)
    # print(dim_reduction.explained_variance_ratio_)
    # print(X.shape)
    X, channels, labels = load_cytof()
    # data = datasets.load_breast_cancer()
    # data = datasets.load_digits()
    # X = data.data
    # labels = data.target
    # data = datasets.fetch_olivetti_faces()
    # dim_reduction = PCA(100)
    # X = dim_reduction.fit_transform(data.data)
    # print(dim_reduction.explained_variance_ratio_)
    # labels = data.target
    # A, gt, coords = sbm(1000, 2, 0, .5, .1)
    # plt.figure()
    # plt.scatter(coords[:, 0], coords[:, 1], c = gt)
    # plt.show()

    tree = TreeSNE(init_df = 1, df_ratio = .8, perp = None, map_dims = 1, late_exag_coeff = 10, dynamic_perp = True, init_with_pca = False, max_iter = 1000)
    # use .8 for bio thing
    # and .7 for MNIST
    # clusters = tree._get_tsne_clusters_via_pop_off(data.data, 1)
    embeddings = tree.fit(X, n_layers = 25)
    np.save("embeddings.npy", embeddings)
    # print(sum(np.isclose(np.sort(embeddings[:, 0], axis = 0), np.sort(embeddings[:, 1], axis = 0))))
    # print(np.sort(embeddings[:, 0], axis = 0)[:10])
    # print(np.sort(embeddings[:, 1], axis = 0)[:10])
    # plt.figure()
    # plt.scatter(embeddings[:, 0], np.zeros(embeddings.shape[0]), c = data.target)
    # plt.figure()
    # plt.scatter(embeddings[:, 1], np.zeros(embeddings.shape[0]), c = data.target)
    # plt.show()
    print(embeddings.shape)
    display_tree_categorical(embeddings, labels)
