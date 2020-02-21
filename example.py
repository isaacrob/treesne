import sys; sys.path.append('/path/to/FIt-SNE')

from tree_sne import TreeSNE
from display_tree import display_tree_mnist
import numpy as np
import scipy.io
from sklearn.metrics import normalized_mutual_info_score
# to reproduce our sample image
SEED = 103
np.random.seed(SEED)

# load the USPS sample dataset
data = scipy.io.loadmat("USPS.mat")
X = data['fea']
X = X.reshape(X.shape[0], -1)
labels = data['gnd'].reshape(-1) - 1

# seed the tree with the random seed
tree = TreeSNE(rand_state = SEED)
# fit the tree with 30 layers
embeddings, layer_clusters, best_clusters = tree.fit(X, n_layers = 30)

# get NMI
nmi = normalized_mutual_info_score(best_clusters, labels, 'geometric')
print("\n\n\n\n-----------------------------------\n\n\n\n")
print("got an NMI of %f on USPS"%(nmi))

# display the tree
display_tree_mnist(embeddings, true_labels = labels, legend_labels = list(np.unique(labels)), distinct = True)