import numpy as np


# taking this from ML for Bio pset 2

def sbm(N, k, pij, pii, sigma):
    """sbm: Construct a stochastic block model

    Args:
        N (integer): Graph size
        k (integer): Number of clusters
        pij (float): Probability of intercluster edges
        pii (float): probability of intracluster edges

    Returns:
        A (numpy.array): Adjacency Matrix
        gt (numpy.array): Ground truth cluster labels
        coords(numpy.array): plotting coordinates for the sbm
    """
    # adjacency matrices are symmetric so only need n(n+1)/2 random numbers
    # then make into upper triangular matrix
    # then deal with and then copy upper triangle to lower triangle
    # A = np.full((N, N), False)
    A = np.zeros((N, N))
    values = np.random.uniform(0, 1, N*(N+1)//2)
    triangular_indices = np.triu_indices(N)
    # print(triangular_indices)
    # print(values)
    A[triangular_indices] = values
    # print(A)
    # print((A != 0).astype(int))

    gt = np.zeros(N)
    # we'll add the coordinates of the centers for each cluster later
    # for now just need offsets from centers
    coords = np.random.normal(loc = 0, scale = sigma, size = (N, 2))
    timesteps = np.linspace(0, 2*np.pi * k / (k + 1), k)
    centers = np.empty((k, 2))
    centers[:, 0] = np.sin(timesteps)
    centers[:, 1] = np.cos(timesteps)

    cluster_spacings = np.linspace(0, N, k + 1).round().astype(int)
    cluster_ends = zip(cluster_spacings[:-1], cluster_spacings[1:])
    for i, (start, end) in enumerate(cluster_ends):
        # doing the assignment this way also captures some in the 
        # lower triangle, but it's not too bad of an approach
        # and easy to understand
        # could maybe do something faster by indexing into triangular_indices
        # but might make keep track of the correct coordinates in parallel harder
        A[start:end, start:end] = A[start:end, start:end] < pii
        A[start:end, end:] = A[start:end, end:] < pij
        coords[start:end] += centers[i]
        gt[start:end] = i

    lower_triangular_indices = np.tril_indices(N, -1)
    A[lower_triangular_indices] = A.T[lower_triangular_indices]
    # then quickly convert to int because that's
    # what the assignment wants
    A = A.astype(int)
    # now make sure it's symmetric
    # assert np.all(A == A.T), "not symmetric"

    return A, gt, coords