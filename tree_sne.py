from fast_tsne import fast_tsne # this is defined in PYTHONPATH
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from scipy.linalg import eigh
from collections import Counter

SEED = 37
ZERO_CUTOFF = 1e-10
np.random.seed(SEED)

class TreeSNE():
    # grow a tree by initializing (1D) embedding
    # then modifying embedding with smaller and smaller df
    # returns matrix with points and different 1D embedding
    # locations
    # later, return matrix containing cluster level assignments
    def __init__(self, init_df = 1, df_ratio = None, rand_state = SEED, ignore_later_exag = True, map_dims = 1, perp = None, late_exag_coeff = 12, dynamic_perp = True, max_iter = 1000, knn_algo = "vp-tree", theta = .5, dynamic_df = True, conservativeness = 2):
        self.init_df = init_df
        # if df_ratio is None, will be defined automatically later based on number of layers
        self.df_ratio = df_ratio
        self.rand_state = rand_state
        # assert ignore_later_exag is False, "ignore_later_exag not yet supported"
        self.ignore_later_exag = ignore_later_exag
        self.map_dims = map_dims
        self.perp = perp
        self.late_exag_coeff = late_exag_coeff
        self.max_iter = max_iter
        assert knn_algo in ['vp-tree', 'annoy'], 'knn algo unsupported, should be one of vp-tree or annoy'
        self.knn_algo = knn_algo
        self.theta = theta
        self.conservativeness = conservativeness

        self.dynamic_perp = dynamic_perp
        self.dynamic_df = dynamic_df
        if dynamic_perp:
            self.curr_perp = self.perp

        self.curr_df = self.init_df

        self._subsample_inds = None

    def _grow_tree_once(self, X, init_embed):
        if self.dynamic_df:
            self.curr_df *= self.df_ratio

        if self.dynamic_perp:
            self.curr_perp = self.curr_perp ** self.df_ratio
            # self.curr_perp = 1 + self.curr_perp * self.df_ratio

        perp = self.perp if not self.dynamic_perp else self.curr_perp

        new_embed = fast_tsne(
            X,
            map_dims = self.map_dims,
            perplexity = perp,
            df = self.curr_df, 
            initialization = init_embed, 
            seed = self.rand_state, # I don't think this is necessary ..
            # load_affinities = "load",
            stop_early_exag_iter = 0 if self.ignore_later_exag else 250, # 250 is default value in library,
            start_late_exag_iter = 0 if self.late_exag_coeff != -1 else -1,
            late_exag_coeff = self.late_exag_coeff,
            learning_rate = X.shape[0] / self.late_exag_coeff,
            knn_algo = self.knn_algo,
            search_k = 150 * int(perp),
            max_iter = self.max_iter,
            theta = self.theta, 
            # nbody_algo = "Barnes-Hut",
            # nterms = 10,
            # min_num_intervals = 50,
        )

        return new_embed

    def _grow_tree(self, X, n_layers = 30, get_clusters = True, bottom = .01):
        if self.df_ratio is None:
            # set by default to scale down to df .01 in the number of layers
            self.df_ratio = 2**(np.log2(bottom) / n_layers)
            print("using df_ratio %f to reach df .01 in %d layers"%(self.df_ratio, n_layers))

        if self.perp is None:
            self.perp = X.shape[0] ** .5
            self.curr_perp = self.perp

        print("getting embedding 1")

        new_embed = fast_tsne(
            X,
            map_dims = self.map_dims, 
            perplexity = self.perp,
            df = self.init_df,
            # initialization = None if not self.init_with_pca else init_embed,
            seed = self.rand_state,
            # load_affinities = "save",
            start_late_exag_iter = 0 if self.late_exag_coeff != -1 else -1,
            late_exag_coeff = self.late_exag_coeff,
            learning_rate = X.shape[0] / self.late_exag_coeff,
            knn_algo = self.knn_algo,
            search_k = 150*int(self.perp),
            max_iter = self.max_iter,
            theta = self.theta,
            # nbody_algo = "Barnes-Hut", 
            # nterms = 10,
            # min_num_intervals = 50,
        )

        embeddings = [new_embed]
        for i in range(n_layers - 1):
            print("getting embedding %d"%(i + 2))
            new_embed = self._grow_tree_once(X, new_embed)
            embeddings.append(new_embed)

        if get_clusters:
            old_k = 0
            best_k = 1
            # best_level = 0
            top_df = self.init_df
            curr_df = self.init_df
            best_df_range = 0
            # best_n_dups = 0
            # n_dups = 0
            clusters = []
            best_clusters = np.zeros(X.shape[0])
            for i, new_embed in enumerate(embeddings):
                print("clustering level %d"%(i))
                this_clusters, k = self._get_clusters_via_subsampled_spectral(new_embed)
                # this_clusters, k = self._get_clusters_via_block_spectral(new_embed)
                this_clusters = self._convert_labels_to_increasing(this_clusters, new_embed, k)
                # print(this_clusters[:20])
                if k <= old_k:
                    # n_dups += 1
                    print("duplicate at %d clusters"%old_k)
                    clusters.append(clusters[-1])
                    df_range = top_df - curr_df
                    if df_range > best_df_range and old_k != 1: # only consider valid clustering after first split
                        print("found new best clustering with k=%d"%old_k)
                        best_df_range = df_range
                        # if best_k != old_k:
                        #     # need to update the best level
                        #     best_level = i
                        best_k = old_k
                        best_clusters = clusters[-1]
                    print(df_range)
                else:
                    n_dups = 0
                    top_df = curr_df
                    old_k = k
                    clusters.append(this_clusters)
                print(old_k)
                # print(k)
                curr_df *= self.df_ratio
            clusters = np.array(clusters)

        embeddings = np.array(embeddings).reshape(X.shape[0], len(embeddings), -1)

        if get_clusters:
            # print(best_clusters)
            if best_df_range == -1:
                print("no stable clustering found, try using more levels")
            else:
                print("best clustering had %d clusters, with df range %f"%(best_k, best_df_range))
            return embeddings, clusters, best_clusters
        else:
            return embeddings

    def _get_snn_graph(self, samples, n_neighbors, fix_lonely = False):
        nn_graph = NearestNeighbors(n_neighbors = n_neighbors).fit(samples.reshape(-1, 1))
        A = nn_graph.kneighbors_graph(samples.reshape(-1, 1))
        A = A - np.eye(A.shape[0])
        A = np.floor((A + A.T) / 2) # if floored, is shared nearest neighbors
        # take ones where they have no neighbors and add only one neighbor, which can't mess up clusters
        if fix_lonely:
            are_you_lonely = np.array(A.sum(axis = 1) == 0).flatten()
            # print(are_you_lonely.shape)
            if np.sum(are_you_lonely) != 0:
                # print("found lonely points")
                lonely_individuals = samples[are_you_lonely].reshape(-1, 1)
                # print(lonely_individuals.shape)
                lonely_friends = nn_graph.kneighbors_graph(lonely_individuals)
                # print(type(lonely_friends))
                # print(lonely_friends.shape)
                for i, these_friends in enumerate(lonely_friends.A):
                    these_friends = these_friends.reshape(-1)
                    # print(these_friends.shape)
                    places_to_check = np.logical_and(these_friends == 1, ~are_you_lonely)
                    friends = np.where(places_to_check)[0]
                    friend_dists = np.abs(lonely_individuals[i] - samples[friends])
                    best_friend_index = np.argmin(friend_dists)
                    best_friend = friends[best_friend_index]
                    # print(A.shape)
                    A[are_you_lonely, best_friend] = 1
                    A[best_friend, are_you_lonely] = 1
            # A[are_you_lonely, are_you_lonely] = lonely_friends
        # then do it again to re symmetrize
        A = (A + A.T) / 2
        assert np.sum(A != A.T) == 0, "A is not symmetric"
        if fix_lonely:
            assert np.sum(A.sum(axis = 1) == 0) == 0, "someone is lonely"
        assert np.sum(A < 0) == 0, "an entry is negative"
        sums = np.array(A.sum(axis = 1)).reshape(-1)
        D = np.diag(sums)
        L = D - A
        # L = (L + L.T) / 2
        # D_shrunken = np.diag(sums**(-1/2))
        # L = D_shrunken@L@D_shrunken

        return L

    def _get_clusters_via_subsampled_spectral(self, X, n_neighbors = None, n_samples = 2000):
        n_samples = min(n_samples, X.shape[0])
        if n_neighbors is None:
            # use number of neighbors based off Erdos graph disconnection criteria of 2logn
            n_neighbors = int(self.conservativeness*np.int(np.log2(n_samples)))
        X = np.array(X)
        # sample_count = X.shape[0]
        if self._subsample_inds is None:
            # consider writing a more intelligent subsampling technique here
            # PHATE uses spectral clustering to find optimal subsampled things
            self._subsample_inds = np.random.choice(X.shape[0], n_samples, replace = False)
        samples = X[self._subsample_inds]
        L = self._get_snn_graph(samples, n_neighbors, fix_lonely = False)
        eig_val, eig_vec = eigh(L, eigvals = [0, int(X.shape[0] ** .5)])
        # eig_val = eig_val
        signals = eig_val < ZERO_CUTOFF
        k = sum(signals)

        indicators = np.array(eig_vec[:, :k]) # shouldn't have to do abs ?
        _, clusters = np.unique(indicators.astype(np.float16), return_inverse = True, axis = 0)
        
        classifier = KNeighborsClassifier(n_neighbors = n_neighbors).fit(samples, clusters)
        all_clusters = classifier.predict(X).astype(np.int)

        # note that KNN doesn't always assign all labels
        # so need to reassign k
        k = len(np.unique(all_clusters))

        return all_clusters, k

    def _convert_labels_to_increasing(self, labels, embedding, k):
        # print(embedding.shape)
        inds = np.argsort(embedding, axis = 0).flatten()
        # print(inds.shape)
        # print(inds[:20])
        label_dict = dict()
        n_added = 0
        for label in labels[inds]:
            if label not in label_dict:
                label_dict[label] = n_added
                n_added += 1
                # if n_added > k:
                #     break

        new_labels = []
        for label in labels:
            new_labels.append(label_dict[label])

        new_labels = np.array(new_labels)

        # assert np.sum(np.diff(new_labels[inds]) < 0) == 0, "labels are not sorted" 

        return new_labels

    def fit(self, *args, **kwargs):
        # just wraps self._grow_tree
        return self._grow_tree(*args, **kwargs)

