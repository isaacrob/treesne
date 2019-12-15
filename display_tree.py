import matplotlib.pyplot as plt
import numpy as np

def display_tree(embeddings, true_labels = None):
    plt.figure()
    embeddings = embeddings.reshape(embeddings.shape[1], -1)
    # print(embeddings.shape)
    for i, embedding in enumerate(embeddings):
        # embedding = embedding[true_labels == 0]
        # print(embedding.shape)
        if true_labels is not None:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = .05, c = true_labels)
            color_bar = plt.colorbar()
            color_bar.set_alpha(1)
            color_bar.draw_all()
        else:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = .05)
    plt.show()

def display_tree_categorical(embeddings, true_labels = None):
    x = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1])
    y = np.repeat(np.arange(embeddings.shape[1]), embeddings.shape[0])
    large_labels = np.tile(true_labels, embeddings.shape[1])
    unique_labels = set(true_labels)
    for val in unique_labels:        
        em = x[large_labels == val]
        y_val = y[large_labels == val]
        plt.plot(em, y_val, alpha = .05, marker='o', linestyle='', label = val)
    
    handles, pltlabels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(pltlabels, handles))
    by_label = {label:handles[i] for i, label in enumerate(pltlabels)}
    legend = plt.legend(by_label.values(), by_label.keys())
    for lh in legend.legendHandles: 
        lh._legmarker.set_alpha(1)
    #plt.legend()
    plt.show()