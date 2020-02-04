import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def getColor(c, N, idx, distinct = False):
    """
    Source for continuous colormap part: https://stackoverflow.com/questions/45612129/cdf-matplotlib-not-enough-colors-for-plot-python
    c is the name of the colormap 
        (see https://matplotlib.org/examples/color/colormaps_reference.html for a list)
    N is the number of colors you want in total
    idx is just an index that will yield the specific color
    """
    distinct_colors_long = ["#004D43", "#0089A3", "#1CE6FF", "#FF34FF", "#FF4A46", \
                            "#008941", "#006FA6", "#A30059", "#72418F", "#FF913F", \
                            "#0000A6", "#63FFAC", "#B79762", "#FFBE06", "#5A0007", \
                            "#997D87", "#CB7E98", "#3A2465", "#324E72", "#A4E804", \
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
                            "#201625", "#BC23FF", "#99ADC0", "#FFFF00", \
                            "#922329", "#5B4534", "#404E55", "#FFDBE5"] #7A4900 "#809693"=gray
    if distinct and N <= len(distinct_colors_long):
        # distinct_colors_26 = np.asarray([(240,163,255),(0,117,220),(153,63,0),(76,0,92),(25,25,25), \
        #                         (0,92,49),(43,206,72),(255,204,153),(128,128,128),(148,255,181), \
        #                         (143,124,0),(157,204,0),(194,0,136),(0,51,128),(255,164,5), \
        #                         (255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143), \
        #                         (224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0), \
        #                         (255,80,5)]) / 255

        
        return distinct_colors_long[idx]

    else:
        import matplotlib as mpl
        cmap = mpl.cm.get_cmap(c)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
        return cmap(norm(idx))



def display_tree(embeddings, true_labels = None, level_labels = None, transparency = None):
    dotsize = 10
    if transparency is None:
        if true_labels is None:
            transparency = 0.05
        else:
            transparency = 300/float(len(true_labels))

    plt.figure()
    embeddings = embeddings.reshape(embeddings.shape[1], -1)
    # print(embeddings.shape)
    for i, embedding in enumerate(embeddings):
        # embedding = embedding[true_labels == 0]
        # print(embedding.shape)
        if true_labels is not None:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = transparency, c = true_labels, s = dotsize)
        elif level_labels is not None:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = transparency, c = level_labels[i], s = dotsize)
        else:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = transparency, s = dotsize)
    if true_labels is not None:
        color_bar = plt.colorbar()
        color_bar.set_alpha(1)
        color_bar.draw_all()
    plt.show()

def display_tree_categorical(embeddings, true_labels, legend_labels = None, transparency = None, distinct = False, not_gray = None):
    """
    Method to plot tree if labels are categorical rather than continuous
    Parameters:
    * embeddings : numpy array of embeddings from TreeSnee.fit()
    * true_labels : numpy 1D array of labels for each observation
    * legend_labels (optional) : list-like object containing labels to include in plot 
                                 Note that observations with labels not in this list will not be plotted
                                 Provide this argument to fix the order of legend labels
    * transparency (optional) : float in range [0,1] specifying transparency (== alpha parameter for plotting)
                                default: 500 / number of observations
    * distinct (optional) : boolean : True => get colors from list of ~58 distinct colors
                                      False => get colors from sampling from a continous color map (viridis)
    * not_gray (optional) : list-like object specifying labels to color - observations with any other 
                            label will be gray. only non-gray labels will be included in the legend
                            when legend_labels is also given, not_gray must be a subset of legend_labels
                            and not_gray will override legend_labels. 
                            specify both if consistency of colors across multiple plots is needed
    """
    dotsize = 10
    if transparency is None:
        transparency = 300/float(len(true_labels))

    x = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1])
    y = np.repeat(np.arange(embeddings.shape[1]), embeddings.shape[0])
    large_labels = np.tile(true_labels, embeddings.shape[1])
    if legend_labels is None:
        legend_labels = list(set(true_labels))

    num_colors = len(legend_labels)
    
    col_index = 0
    not_gray_colors = {}
    for val in legend_labels:
        transparency_use = transparency
        if not_gray is None:
            c = getColor("viridis", num_colors, col_index, distinct)
        elif val in not_gray:
            c = getColor("viridis", num_colors, col_index, distinct)
            not_gray_colors[val] = c
            transparency_use = 0 # will plot for real later
        else: # val is not in not_gray but not_gray is not None --> make it gray
            c = "#bebebe" # gray it out bebe
        
        em = x[large_labels == val]
        y_val = y[large_labels == val]
        plt.plot(em, y_val, alpha = transparency_use, marker='o', linestyle='', label = val,
                color = c)#, s = dotsize)
        col_index += 1

    if not_gray is not None:
        for val in not_gray:
            c = not_gray_colors[val]
            em = x[large_labels == val]
            y_val = y[large_labels == val]
            plt.plot(em, y_val, alpha = transparency, marker='o', linestyle='', color = c)#, s = dotsize)

    
    handles, pltlabels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(pltlabels, handles))
    if not_gray is None:
        # by_label = {label:handles[i] for i, label in enumerate(pltlabels)}
        legend = plt.legend(by_label.values(), by_label.keys())
    else:
        by_label_not_gray = {label:by_label[label] for label in not_gray}
        legend = plt.legend(by_label_not_gray.values(), by_label_not_gray.keys())
    for lh in legend.legendHandles: 
        lh._legmarker.set_alpha(1)
    #plt.legend()
    plt.show()



def display_tree_mnist(embeddings, true_labels = None, transparency = None, legend_labels = None, numeric_labels=True, distinct=False):
    dotsize = 10
    if transparency is None:
        if true_labels is None:
            transparency = 0.05
        else:
            transparency = 300/float(len(true_labels))

    if distinct or not numeric_labels:
        colordict = {}
        num_colors = len(legend_labels)
        for i in range(num_colors):
            colordict[legend_labels[i]] = getColor("viridis", num_colors, i, distinct=True)

    plt.figure()
    embeddings = embeddings.reshape(embeddings.shape[1], -1)
    # print(embeddings.shape)
    for i, embedding in enumerate(embeddings):
        # embedding = embedding[true_labels == 0]
        # print(embedding.shape)
        if true_labels is not None and numeric_labels and not distinct:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = transparency, c = true_labels, s = dotsize)
        elif true_labels is not None and (distinct or not numeric_labels):
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = transparency, 
                            c = [colordict[x] for x in true_labels], s = dotsize)
        else:
            plt.scatter(embedding, np.ones(embedding.shape[0]) * i, alpha = transparency, s = dotsize)
    if true_labels is not None:
        if legend_labels is not None:
            legend_elems = []
            num_colors = len(legend_labels)
            for i in range(num_colors):
                label = legend_labels[i]
                if distinct or not numeric_labels:
                    color = getColor("viridis", num_colors, i, distinct=True)
                else:
                    color = getColor("viridis", num_colors, i)
                legend_elems.append(Line2D([0],[0], marker = 'o', alpha=1, color = 'w',
                                            markerfacecolor=color, label = label))
            legend = plt.legend(handles=legend_elems)


        else: 
            color_bar = plt.colorbar()
            color_bar.set_alpha(1)
            color_bar.draw_all()
    plt.show()





