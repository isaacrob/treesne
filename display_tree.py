import matplotlib.pyplot as plt
import numpy as np

def getColor(c, N, idx):
    """
    Source: https://stackoverflow.com/questions/45612129/cdf-matplotlib-not-enough-colors-for-plot-python
    c is the name of the colormap 
        (see https://matplotlib.org/examples/color/colormaps_reference.html for a list)
    N is the number of colors you want in total
    idx is just an index that will yield the specific color
    """
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))

def display_tree(embeddings, true_labels = None):
    plt.figure()
    embeddings = embeddings.reshape(embeddings.shape[1], -1)
    # print(embeddings.shape)
    for i, embedding in enumerate(embeddings):
        # embedding = embedding[true_labels == 0]
        # print(embedding.shape)
        plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = .05, c = true_labels)
    color_bar = plt.colorbar()
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.show()

def display_tree_categorical(embeddings, true_labels, legend_labels = None):
    """
    Method to plot tree if labels are categorical rather than continuous
    """
    x = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1])
    y = np.repeat(np.arange(embeddings.shape[1]), embeddings.shape[0])
    large_labels = np.tile(true_labels, embeddings.shape[1])
    if legend_labels is None:
        legend_labels = set(true_labels)

    num_colors = len(legend_labels)
    
    col_index = 0
    for val in legend_labels:        
        em = x[large_labels == val]
        y_val = y[large_labels == val]
        plt.plot(em, y_val, alpha = .05, marker='o', linestyle='', label = val,
                color = getColor("viridis", num_colors, col_index))
        col_index += 1
    
    handles, pltlabels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(pltlabels, handles))
    by_label = {label:handles[i] for i, label in enumerate(pltlabels)}
    legend = plt.legend(by_label.values(), by_label.keys())
    for lh in legend.legendHandles: 
        lh._legmarker.set_alpha(1)
    #plt.legend()
    plt.show()