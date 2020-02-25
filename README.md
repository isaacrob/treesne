# tree-SNE

t-SNE and hierarchical clustering are popular methods of exploratory data analysis, particularly in biology. Building on recent advances in speeding up t-SNE and obtaining finer-grained structure, we combine the two to create tree-SNE, a hierarchical clustering and visualization algorithm based on stacked one-dimensional t-SNE embeddings. We also introduce alpha-clustering, which recommends the optimal cluster assignment, without foreknowledge of the number of clusters, based off of the cluster stability across multiple scales. We demonstrate the effectiveness of tree-SNE and alpha-clustering on images of handwritten digits, mass cytometry (CyTOF) data from blood cells, and single-cell RNA-sequencing (scRNA-seq) data from retinal cells. Furthermore, to demonstrate the validity of the visualization, we use alpha-clustering to obtain unsupervised clustering results competitive with the state of the art on several image data sets.

ArXiv preprint: https://arxiv.org/abs/2002.05687

## Prerequisites

Install Fit-SNE from https://github.com/KlugerLab/FIt-SNE and add the FIt-SNE directory that you cloned to your PYTHONPATH environmental variable. This lets tree-SNE access the Python file used to interface with FIt-SNE. This can be done one of several ways:
- run `export PYTHONPATH="$PYTHONPATH":/path/to/FIt-SNE` in your terminal before running your Python script using tree-SNE
- add `export PYTHONPATH="$PYTHONPATH":/path/to/FIt-SNE` to your .bash_profile
- add the line `import sys; sys.path.append('/path/to/FIt-SNE/')` to your Python script before calling `import tree_sne`

Also make sure to have Numpy, Scipy, Sklearn, and Matplotlib installed.

We've tested with Python 3.6+.

## Test/Example

Run `example.py` to make sure everything is set up right. This will run tree-SNE on the USPS handwritten digit dataset, run alpha-clustering, calculate the NMI, and display the tree. You can refer to this file for calling conventions. Note the top line adding FIt-SNE to the Python path.

## Sample Usage

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

If your data has more clusters, reduce the `conservativeness` parameter to `TreeSNE`. Typical values range from 1 to 2. It should never drop below 1 according to our theory motivation for its implementation, and we've only had to decrease it when trying to find 100 clusters, in which case we set it to 1.3. `n_layers` and `conservativeness` are the only two parameters that we think users may want to adjust, at least for the time being. Once we've refactored we'll write more documentation. Note that `conservativeness` only effects alpha-clustering and does not actually change the tree-SNE embedding itself.

![MNIST tree-SNE example plot](https://i.imgur.com/I5iYOEj.png)

## Authors

* **Isaac Robinson** - isaac.robinson@yale.edu
* **Emma Pierce-Hoffman** - emma.pierce-hoffman@yale.edu


## Acknowledgments

The authors thank Stefan Steinerberger for inspiration, support, and advice; George Linderman for enabling one-dimensional t-SNE with degrees of freedom < 1 in the FIt-SNE package; Scott Gigante for data pre-processing and helpful discussions of visualizations and alpha-clustering; Smita Krishnaswamy for encouragement and feedback; and Ariel Jaffe for discussing the Nyström method and its relationship to subsampled spectral clustering.
