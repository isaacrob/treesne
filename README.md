# tree-SNE

t-SNE and hierarchical clustering are popular methods of exploratory data analysis, particularly in biology. Building on recent advances in speeding up t-SNE and obtaining finer-grained structure, we combine the two to create tree-SNE, a hierarchical clustering and visualization algorithm based on stacked one-dimensional t-SNE embeddings. We also introduce alpha-clustering, which recommends the optimal cluster assignment, without foreknowledge of the number of clusters, based off of the cluster stability across multiple scales. We demonstrate the effectiveness of tree-SNE and alpha-clustering on images of handwritten digits, mass cytometry (CyTOF) data from blood cells, and single-cell RNA-sequencing (scRNA-seq) data from retinal cells. Furthermore, to demonstrate the validity of the visualization, we use alpha-clustering to obtain unsupervised clustering results competitive with the state of the art on several image data sets.

## Prerequisites

Install Fit-SNE from https://github.com/KlugerLab/FIt-SNE and add the fast_tsne.py to your Python path. Also make sure to have Numpy, Scipy, Sklearn, and Matplotlib installed.

## Example Usage

Assuming you have a 2D Numpy array containing your data in a variable `X`. To build a tree-SNE plot with 30 layers, cluster on each layer, and determine the optimal clustering via alpha-clustering (note does not require preknowledge of the number of clusters):

	from tree_sne import TreeSNE

	tree = TreeSNE()
	embeddings, layer_clusters, best_clusters = tree.fit(X, n_layers = 30)

The `embeddings` variable will contain each data point's embedding in each layer, with `embeddings.shape` of (n_points, n_layers, n_features). For now, n_features will always be 1, as we haven't yet implemented stacked 2D t-SNE embeddings. The variable `layer_clusters` will contain cluster assignments for each point in each layer of the embedding, and `best_clusters` will contain optimal cluster assignments for the data.

To display the tree using our code with cluster labels, run:

	from display_tree import display_tree_mnist
	import numpy as np

	display_tree_mnist(embeddings, true_labels = best_clusters, legend_labels = list(np.unique(best_clusters)), distinct = True)

Alternatively, some labels you provide can be used instead of `best_clusters`. We realize this is messy but until we refactor this is what we have. We're sorry. You don't have to use our display code if you don't want to, and we'll improve it soon.

An example pipeline loading MNIST from Tensorflow, generating a 30 layer tree-SNE embedding, and then displaying that embedding with our display function:

	import tensorflow as tf
	from tree_sne import TreeSNE
	from display_tree import display_tree_mnist
	from sklearn.decomposition import PCA
	import numpy as np
	# to reproduce our sample image
	SEED = 42
	np.random.seed(42)

	# load MNIST from Tensorflow
	(Xt, tlabels), (X, labels) = tf.keras.datasets.mnist.load_data()
	# normalize and reshape
	Xt, X = Xt / 255.0, X / 255.0
	X = X.reshape(X.shape[0], -1)

	# make sure the labels match their digits, not classes
	# since tf has them in 1..10 format but we want 0..9
	labels -= 1

	# reduce dimension to speed up embedding
	X = PCA(100).fit_transform(X)

	# seed the tree with the random seed
	tree = TreeSNE(rand_state = SEED)
	# fit the tree with 30 layers
	embeddings, layer_clusters, best_clusters = tree.fit(X, n_layers = 30)

	# display the tree
	display_tree_mnist(embeddings, true_labels = labels, legend_labels = list(np.unique(labels)), distinct = True)

If your data has more clusters, reduce the `conservativeness` parameter to `TreeSNE`. Typical values range from 1 to 2. It should never drop below 1 according to our theory motivation for its implementation, and we've only had to decrease it when trying to find 100 clusters, in which case we set it to 1.3. `n_layers` and `conservativeness` are the only two parameters that we think users may want to adjust, at least for the time being. Once we've refactored we'll write more documentation. Note that `conservativeness` only effects alpha-clustering and does not actually change the tree-SNE embedding itself.

![MNIST tree-SNE example plot]
(https://i.imgur.com/4YjJGgM.png)

## Authors

* **Isaac Robinson** - isaac.robinson@yale.edu
* **Emma Pierce-Hoffman** - emma.pierce-hoffman@yale.edu


## Acknowledgments

The authors thank Stefan Steinerberger for inspiration, support, and advice; George Linderman for enabling one-dimensional t-SNE with degrees of freedom < 1 in the FIt-SNE package; Scott Gigante for data pre-processing and helpful discussions of visualizations and alpha-clustering; Smita Krishnaswamy for encouragement and feedback; and Ariel Jaffe for discussing the Nyström method and its relationship to subsampled spectral clustering.
