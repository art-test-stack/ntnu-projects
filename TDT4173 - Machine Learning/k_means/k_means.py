import numpy as np 
import pandas as pd
from scipy.stats import multivariate_normal

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

versions = ['v1', 'v2', 'v3', 'em']

class KMeans:
    
    def __init__(self, K: int = 2, version: str = 'v1'):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        assert version in versions, f"'{version}' is not a version of this program"
        self.K = K
        self.version = version
        
        # NOTE: this data are not useful for the algorithm, 
        # it increase it complexity but it permit to get 
        # the plots for the report
        self.training_distortions_to_plot = []
        self.clusters_to_plot = []

        self.training_distortions = []
        self.training_silhouettes = []

        
    def fit(self, X, nb_epochs: int = 24, loop_tries: int = 0, tolerance: float = 0.01):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        K = self.K
        X = np.array(X)
        
        self.c = np.random.rand(K, X.shape[1])

        match self.version: 
            case 'v1': 
                self.clusters_to_plot.append(self.c)

                for _ in range(nb_epochs):
                    z = self.predict(X)

                    self.training_distortions_to_plot.append(euclidean_distortion(X, self.predict(X)))

                    r = np.zeros((X.shape[0], self.K))
                    r[np.arange(X.shape[0]), z] = 1
                    
                    if 0. in np.sum(r, axis=0) or 0 in np.sum(r, axis=0):
                        self.c = np.random.rand(K, 2) 
                        continue

                    self.c = (r.T.dot(X).T / sum(r)).T
                    self.clusters_to_plot.append(self.c)
                
                self.training_distortions_to_plot.append(euclidean_distortion(X, self.predict(X)))

            case 'v2':
                clusters = []
                
                for _ in range(loop_tries):

                    _model = KMeans(K=K, version='v1')
                    _model.fit(X=X, nb_epochs=nb_epochs)

                    self.clusters_to_plot.append(_model.clusters_to_plot)

                    self.training_distortions_to_plot.append(_model.training_distortions_to_plot)
                    self.training_distortions.append(euclidean_distortion(X, _model.predict(X)))

                    self.training_silhouettes.append(euclidean_silhouette(X, self.predict(X))) if len(np.unique(self.predict(X))) == K else None

                    clusters.append(_model.c)

                index_of_min_distortion = np.argmin(self.training_distortions)

                self.cluster_to_plot = self.clusters_to_plot[index_of_min_distortion]
                self.c = clusters[index_of_min_distortion]
            
            case 'v3':
                k = 2
                init_model = KMeans(K=k, version='v1')
                init_model.fit(X)
                z = init_model.predict(X)

                silhouette = euclidean_silhouette(X, z)

                last_silhouette = silhouette

                last_clusters = init_model.c
                clusters = init_model.c

                self.training_distortions.append(euclidean_distortion(X, z)) 
                self.training_silhouettes.append(silhouette)

                while silhouette >= last_silhouette :
                    k += 1
                    last_clusters = clusters
                    last_silhouette = silhouette

                    _model = KMeans(K=k, version='v2')
                    _model.fit(X=X, nb_epochs=nb_epochs, loop_tries=loop_tries)

                    silhouette = euclidean_silhouette(X, _model.predict(X))
                    clusters = _model.c
                    
                    self.training_distortions.append(euclidean_distortion(X, _model.predict(X))) if len(np.unique(_model.predict(X))) == k else None
                    self.training_silhouettes.append(silhouette)


                self.K = k
                self.c = last_clusters

            case 'em':
                K = self.K
                n, p = X.shape

                self.c = np.random.rand(K, p)
                pis = (K / n * np.ones(K))

                init_sigs = [ np.random.rand(K, p) for _ in range(K)] * 10
                covs = np.array([ init_sigs[k].T.dot(init_sigs[k]) for k in range(K)])

                last_likelihood = 0

                self.likelihoods = []

                for _ in range(nb_epochs):

                    # E step
                    weights = np.zeros((K, n))
                    for k in range(K):
                        weights[k, :] = pis[k] * multivariate_normal(self.c[k], covs[k]).pdf(X)
                    weights /= weights.sum(0)

                    # M step
                    pis = weights.sum(axis=1) / n
                    
                    self.c = weights.dot(X) / weights.sum(1, keepdims=True)

                    covs = np.zeros((K, p, p))
                    for k in range(K):
                        Xc = X - self.c[k, :]
                        covs[k] = (weights[k, :, None, None] * np.matmul(Xc[:, :, None], Xc[:, None, :])
                                   ).sum(axis=0) / weights.sum(axis=1)[k, None, None]

                        
                    likelihood = 0
                    for pi, c, cov in zip(pis, self.c, covs):
                        likelihood += pi * multivariate_normal(c, cov).pdf(X)

                    likelihood = np.log(likelihood).sum()

                    if np.abs(likelihood - last_likelihood) < tolerance:
                        break
                    

                    self.likelihoods.append(np.abs(likelihood - last_likelihood))
                    last_likelihood = likelihood


                print('final likelihood diff: ', likelihood - last_likelihood)
                self.pis = pis
                self.covs = covs
                self.c = self.c
                self.weights = weights


    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = np.array(X)

        if self.version != 'em':
            return np.array([np.argmin(euclidean_distance(x, self.c), axis=0) for x in X], dtype=np.intc)
        
        if self.version == 'em':
            n = X.shape[0]
            weights = np.zeros((self.K, n))
            for k in range(self.K):
                weights[k, :] = self.pis[k] * multivariate_normal(self.c[k], self.covs[k]).pdf(X)
            weights /= weights.sum(0)

            return np.argmax(weights, axis=0)

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.c


# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
