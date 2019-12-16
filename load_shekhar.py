import scprep
import h5py
import pandas as pd
import numpy as np

def load_shekhar():
    """
    Method to load Shekhar retinal cell data and save as .npy for easy loading
        Also subsets to data with labels corresponding to provided cell types in cluster_assign
    Adapted from script provided by Scott Gigante
    Assumes pandas < v0.24
    """
    
    # load metadata
    clusters = scprep.io.load_tsv("../shekhar_retinal_bipolar/retina_clusters.tsv")
    cells = scprep.io.load_csv("../shekhar_retinal_bipolar/retina_cells.csv",
                               cell_names=False, gene_names=False)
    genes = scprep.io.load_csv("../shekhar_retinal_bipolar/retina_genes.csv",
                               cell_names=False, gene_names=False)

    # load data matrix
    cells = cells.values.flatten()[:-1] # somehow we're missing one
    genes = genes.values.flatten()
    with h5py.File("../shekhar_retinal_bipolar/retina_data.mat", 'r') as handle:
        data = pd.DataFrame(
            np.array(handle['data']).T,
            index=cells, columns=genes)

    # take intersection of data and metadata
    merged_data = pd.merge(data, clusters, how='left',
                           left_index=True, right_index=True)
    # remove cells without a cluster label -- optional
    merged_data = merged_data.loc[~np.isnan(merged_data['CLUSTER'])]
    # drop the cluster label columns
    data = merged_data.iloc[:,:-2]
    # assign cluster names
    cluster_assign = {
        '1': 'Rod BC',
        '2': 'Muller Glia',
        '7': 'BC1A',
        '9': 'BC1B',
        '10': 'BC2',
        '12': 'BC3A',
        '8': 'BC3B',
        '14': 'BC4',
        '3':  'BC5A',
        '13': 'BC5B',
        '6': 'BC5C',
        '11': 'BC5D',
        '5': 'BC6',
        '4': 'BC7',
        '15_1': 'BC8/9_1',
        '15_2': 'BC8/9_2',
        '16_1':  'Amacrine_1',
        '16_2':  'Amacrine_2',
        '20': 'Rod PR',
        '22': 'Cone PR',
    }
    labels = merged_data.iloc[:,-1].values
    for label, celltype in cluster_assign.items():
        labels = np.where(labels == label, celltype, labels)
        
    keep_inds = [(x in cluster_assign.values()) for x in labels]
    data = data[keep_inds]
    labels = labels[keep_inds]
    
    #np.save("shekhar_data", data.values)
    #np.save("shekhar_labels", labels)

    # return data.values, labels
