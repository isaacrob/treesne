import matplotlib.pyplot as plt
import numpy as np

def getColor(c, N, idx, distinct = False):
    """
    Source for colormap part: https://stackoverflow.com/questions/45612129/cdf-matplotlib-not-enough-colors-for-plot-python
    c is the name of the colormap 
        (see https://matplotlib.org/examples/color/colormaps_reference.html for a list)
    N is the number of colors you want in total
    idx is just an index that will yield the specific color
    """
    if distinct and N <= 26:
        # distinct_colors_26 = np.asarray([(240,163,255),(0,117,220),(153,63,0),(76,0,92),(25,25,25), \
        #                         (0,92,49),(43,206,72),(255,204,153),(128,128,128),(148,255,181), \
        #                         (143,124,0),(157,204,0),(194,0,136),(0,51,128),(255,164,5), \
        #                         (255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143), \
        #                         (224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0), \
        #                         (255,80,5)]) / 255

        distinct_colors_59 = ["#004D43", "#0089A3", "#1CE6FF", "#FF34FF", "#FF4A46", \
                                "#008941", "#006FA6", "#A30059", "#72418F", "#FF913F", \
                                "#0000A6", "#63FFAC", "#B79762", "#FFBE06", "#5A0007", \
                                "#997D87", "#CB7E98", "#809693", "#324E72", "#A4E804", \
                                "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80","#61615A", \
                                "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", \
                                "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", \
                                "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F", \
                                "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", \
                                "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062", \
                                "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", \
                                "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", \
                                "#BEC459", "#456648", "#0086ED", "#886F4C", "#34362D", \
                                "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", \
                                "#8FB0FF", "#938A81", "#575329", "#00FECF", "#B05B6F", \
                                "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", \
                                "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", \
                                "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", \
                                "#201625", "#BC23FF", "#99ADC0", "#3A2465", "#FFFF00", \
                                "#922329", "#5B4534", "#404E55", "#FFDBE5"] #7A4900 
        return distinct_colors_59[idx]

    else:
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
        if true_labels is not None:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = .05, c = true_labels)
        else:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = .05)
    if true_labels is not None:
        color_bar = plt.colorbar()
        color_bar.set_alpha(1)
        color_bar.draw_all()
    plt.show()

def display_tree_categorical(embeddings, true_labels, legend_labels = None, transparency = None, distinct = False):
    """
    Method to plot tree if labels are categorical rather than continuous
    """
    if transparency is None:
        transparency = 500/float(len(true_labels))

    x = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1])
    y = np.repeat(np.arange(embeddings.shape[1]), embeddings.shape[0])
    large_labels = np.tile(true_labels, embeddings.shape[1])
    if legend_labels is None:
        legend_labels = list(set(true_labels))

    num_colors = len(legend_labels)
    
    col_index = 0
    for val in legend_labels:        
        em = x[large_labels == val]
        y_val = y[large_labels == val]
        plt.plot(em, y_val, alpha = transparency, marker='o', linestyle='', label = val,
                color = getColor("viridis", num_colors, col_index, distinct))
        col_index += 1
    
    handles, pltlabels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(pltlabels, handles))
    by_label = {label:handles[i] for i, label in enumerate(pltlabels)}
    legend = plt.legend(by_label.values(), by_label.keys())
    for lh in legend.legendHandles: 
        lh._legmarker.set_alpha(1)
    #plt.legend()
    plt.show()